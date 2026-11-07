// Tiled GEMM generator.
//
// C[z][m][n] = epilogue( alpha * sum_k A(z,m,k) * B(z,n,k) )
//
// A modes : "rows"  A[aOff + z*aBatch + m*lda + k]
//           "conv"  implicit im2col of a batch of NHWC images: m = (b*OH + oy)*OW + ox and
//                   k = (ky*KW + kx)*Cin + ci. Compile-time options:
//                     pad  "zeros" | "reflect"
//                     up   "" | "nearest" | "bilinear": the conv reads its input upsampled 2x
//                          (bilinear with half-pixel centers and clamped edges, as PyTorch/ONNX)
//                     pro  "" | "affine" | "affine_relu": the loaded input becomes x*S + T (then
//                          max(., 0)); padding stays zero. S for (image b, channel c) is
//                          PRO[b*proStride + c] and T is PRO[proOff + b*proStride + c]
//                          (proStride = Cin, or 0 when all images share one row).
//                          "scale": x*S with S = PRO[proOff + b*proStride + c] (a modulated
//                          conv's per-image style, read from a row of every layer's styles).
//                     snake: a 2x2 kernel's taps are stored in the order (0,0), (0,1), (1,1), (1,0)
//                   When Cin % 8 == 0 the 8 k of one load share a tap (vector loads); otherwise
//                   every element is gathered on its own (first layers, Cin = 3 or 28).
// B fmts  : "f32"   B[bOff + z/bDiv*bBatch + n*ldb + k]
//           "bf16"  [N,K] bf16 pairs packed in u32
//           "f16"   [N,K] f16 pairs packed in u32, batched like f32 (Winograd filters)
//           "i8"    [N,K] int8 packed 4 per u32            (per-row scale in epilogue)
//           "w4"    [N,K] 4-bit codes packed 8 per u32, fp8-e4m3 group scale per 16,
//                   16-entry codebook, decoded to the int8 grid (per-row scale in epilogue)
// Epilogue: *rowScale[n], +bias[n], +ADD[m*N + n], act (relu | leaky | prelu | silu | tanh01), then
//           store, or C = C + gate[gOff+n] * v / C = C + v (resid, in place).
// phases  : a transposed conv run as a polyphase 2x2 conv (snake tap order) to 4 phases of
//           P.phases output channels each: a workgroup only runs the k range of the taps its
//           columns' phases use, [0, 4), [1, 3), [2, 4), [2, 3) taps for phases 0..3 (the other
//           taps' weights are zero).
// split   : split-K for problems with too few tiles to fill the GPU. Workgroup z takes the k range
//           [z*kChunk, (z+1)*kChunk) and stores its raw partial sums to C[z*cBatch + m*N + n];
//           splitReduceShader (kernels.js) adds them up and applies the epilogue.
//
// Tiles (256 threads, 16 KB of workgroup memory in both):
//   R = 4: 64x64,   TK = 32, 4x4 outputs per thread  (small problems)
//   R = 8: 128x128, TK = 16, 8x8 outputs per thread  (everything large)
// Every thread stages 8 consecutive k of one A row and one B row per k-tile, and issues the
// next tile's global loads before computing on the current one (register prefetch).
//
// All register arrays are indexed with constants only (the generator unrolls every loop that
// touches them): shader compilers spill dynamically indexed arrays to local memory.

const range = (n) => [...Array(n).keys()];
const cache = new Map();

function loaderRows(vec) {
  if (vec) {
    return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }
  let q = (P.aOff + z * P.aBatch + m * P.lda + k) >> 2u;
  return V8(A[q], A[q + 1u]);
}`;
  }
  const e = range(8).map((j) => `select(0.0, A[base + ${j}u], k + ${j}u < P.K)`);
  return `
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }
  let base = P.aOff + z * P.aBatch + m * P.lda + k;  // out-of-range lanes are masked (robust access keeps reads safe)
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
}

// Input pixel (iy, ix) of the (possibly upsampled) conv input for output (oy, ox) and a tap.
// Returns false when it falls into zero padding.
function convCoords({ pad, up, snake }) {
  const lim = up ? "2u * P.ih" : "P.ih";
  const limw = up ? "2u * P.iw" : "P.iw";
  const bounds = pad === "reflect"
    ? `iy = refl(iy, i32(${lim})); ix = refl(ix, i32(${limw}));`
    : `if (iy < 0 || ix < 0 || iy >= i32(${lim}) || ix >= i32(${limw})) { return false; }`;
  return `
fn refl(i: i32, n: i32) -> i32 { return select(select(i, 2 * n - 2 - i, i >= n), -i, i < 0); }
fn tapPos(oy: u32, ox: u32, stored: u32, pos: ptr<function, vec2<i32>>) -> bool {
  let tap = ${snake ? "select(stored, stored ^ 1u, stored >= 2u)" : "stored"};
  var iy = i32(oy * P.stride + tap / P.kw) - i32(P.pad);
  var ix = i32(ox * P.stride + tap % P.kw) - i32(P.pad);
  ${bounds}
  *pos = vec2<i32>(iy, ix);
  return true;
}`;
}

function loaderConv({ vec, pad, up, pro, snake = false }) {
  const head = `
  let ox = m % P.ow;
  let t = m / P.ow;
  let oy = t % P.oh;
  let b = t / P.oh;`;
  const relu = pro === "affine_relu";
  if (vec) {
    // Cin % 8 == 0: the 8 k share one tap; vector loads of 8 channels
    let sample;
    if (up === "bilinear") {
      sample = `
  let sy = max((f32(p.x) + 0.5) * 0.5 - 0.5, 0.0);
  let sx = max((f32(p.y) + 0.5) * 0.5 - 0.5, 0.0);
  let y0 = u32(sy);
  let x0 = u32(sx);
  let y1 = min(y0 + 1u, P.ih - 1u);
  let x1 = min(x0 + 1u, P.iw - 1u);
  let wy = sy - f32(y0);
  let wx = sx - f32(x0);
  let r0 = (b * P.ih + y0) * P.iw;
  let r1 = (b * P.ih + y1) * P.iw;
  let a00 = ld8((r0 + x0) * P.cin + ci);
  let a01 = ld8((r0 + x1) * P.cin + ci);
  let a10 = ld8((r1 + x0) * P.cin + ci);
  let a11 = ld8((r1 + x1) * P.cin + ci);
  var v = V8(mix(mix(a00.lo, a01.lo, wx), mix(a10.lo, a11.lo, wx), wy), mix(mix(a00.hi, a01.hi, wx), mix(a10.hi, a11.hi, wx), wy));`;
    } else {
      const y = up === "nearest" ? "u32(p.x) / 2u" : "u32(p.x)";
      const x = up === "nearest" ? "u32(p.y) / 2u" : "u32(p.y)";
      sample = `\n  var v = ld8(((b * P.ih + ${y}) * P.iw + ${x}) * P.cin + ci);`;
    }
    const prologue = pro === "scale"
      ? `
  let si = (P.proOff + b * P.proStride + ci) >> 2u;
  v = V8(v.lo * PRO[si], v.hi * PRO[si + 1u]);`
      : pro
        ? `
  let si = (b * P.proStride + ci) >> 2u;
  let so = si + (P.proOff >> 2u);
  v = V8(v.lo * PRO[si] + PRO[so], v.hi * PRO[si + 1u] + PRO[so + 1u]);${relu ? "\n  v = V8(max(v.lo, vec4<f32>(0.0)), max(v.hi, vec4<f32>(0.0)));" : ""}`
        : "";
    return `
fn ld8(i: u32) -> V8 { let q = i >> 2u; return V8(A[q], A[q + 1u]); }
${convCoords({ pad, up, snake })}
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }${head}
  let tap = k / P.cin;
  let ci = k % P.cin;
  var p: vec2<i32>;
  if (!tapPos(oy, ox, tap, &p)) { return V8(); }${sample}${prologue}
  return v;
}`;
  }
  // generic: gather each element
  let sample;
  if (up === "bilinear") {
    sample = `
  let sy = max((f32(p.x) + 0.5) * 0.5 - 0.5, 0.0);
  let sx = max((f32(p.y) + 0.5) * 0.5 - 0.5, 0.0);
  let y0 = u32(sy);
  let x0 = u32(sx);
  let y1 = min(y0 + 1u, P.ih - 1u);
  let x1 = min(x0 + 1u, P.iw - 1u);
  let r0 = (b * P.ih + y0) * P.iw;
  let r1 = (b * P.ih + y1) * P.iw;
  var v = mix(mix(A[(r0 + x0) * P.cin + ci], A[(r0 + x1) * P.cin + ci], sx - f32(x0)),
              mix(A[(r1 + x0) * P.cin + ci], A[(r1 + x1) * P.cin + ci], sx - f32(x0)), sy - f32(y0));`;
  } else {
    const y = up === "nearest" ? "u32(p.x) / 2u" : "u32(p.x)";
    const x = up === "nearest" ? "u32(p.y) / 2u" : "u32(p.y)";
    sample = `\n  var v = A[((b * P.ih + ${y}) * P.iw + ${x}) * P.cin + ci];`;
  }
  const prologue = pro === "scale" ? "\n  v = v * PRO[P.proOff + b * P.proStride + ci];"
    : pro ? `\n  v = v * PRO[b * P.proStride + ci] + PRO[P.proOff + b * P.proStride + ci];${relu ? "\n  v = max(v, 0.0);" : ""}` : "";
  return `
${convCoords({ pad, up, snake })}
fn elemA(b: u32, oy: u32, ox: u32, k: u32) -> f32 {
  if (k >= P.K) { return 0.0; }
  let tap = k / P.cin;
  let ci = k % P.cin;
  var p: vec2<i32>;
  if (!tapPos(oy, ox, tap, &p)) { return 0.0; }${sample}${prologue}
  return v;
}
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }${head}
  return V8(vec4<f32>(${range(4).map((j) => `elemA(b, oy, ox, k + ${j}u)`).join(", ")}),
            vec4<f32>(${range(4).map((j) => `elemA(b, oy, ox, k + ${j + 4}u)`).join(", ")}));
}`;
}

// Element type of the B (weight) binding read by loaderB
export function weightArrayType(b, vec) {
  if (b === "f32") return vec ? "vec4<f32>" : "f32";
  if (b === "bf16" || b === "f16") return vec ? "vec4<u32>" : "u32";
  if (b === "i8") return vec ? "vec2<u32>" : "u32";
  return "u32";
}

// loadB8(z, n, k): 8 consecutive k of weight row n, decoded to f32 (before the per-row scale of
// the quantized formats). Reads P.N, P.K and, for f32/f16, P.bOff, P.bBatch, P.bDiv, P.ldb.
export function loaderB(b, vec) {
  if (b === "f32") {
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let q = (P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k) >> 2u;
  return V8(B[q], B[q + 1u]);
}`;
    }
    const e = range(8).map((j) => `select(0.0, B[base + ${j}u], k + ${j}u < P.K)`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N) { return V8(); }
  let base = P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k;
  return V8(vec4<f32>(${e.slice(0, 4).join(", ")}), vec4<f32>(${e.slice(4).join(", ")}));
}`;
  }
  if (b === "bf16") {
    const lo = (w) => `bitcast<f32>(${w} << 16u)`;
    const hi = (w) => `bitcast<f32>(${w} & 0xffff0000u)`;
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let w = B[(n * P.K + k) >> 3u];
  return V8(vec4<f32>(${lo("w.x")}, ${hi("w.x")}, ${lo("w.y")}, ${hi("w.y")}), vec4<f32>(${lo("w.z")}, ${hi("w.z")}, ${lo("w.w")}, ${hi("w.w")}));
}`;
    }
    // K even; a partial last chunk is zero-masked
    const w = range(4).map((j) => `let w${j} = select(0u, B[base + ${j}u], k + ${2 * j}u < P.K);`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let base = (n * P.K + k) >> 1u;
  ${w.join("\n  ")}
  return V8(vec4<f32>(${lo("w0")}, ${hi("w0")}, ${lo("w1")}, ${hi("w1")}), vec4<f32>(${lo("w2")}, ${hi("w2")}, ${lo("w3")}, ${hi("w3")}));
}`;
  }
  if (b === "f16") {
    // batched (bOff + z*bBatch, in halves), unpack2x16float is core WGSL (no shader-f16 needed)
    if (vec) {
      return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let w = B[(P.bOff + z * P.bBatch + n * P.K + k) >> 3u];
  return V8(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)));
}`;
    }
    const w = range(4).map((j) => `let w${j} = select(vec2<f32>(), unpack2x16float(B[base + ${j}u]), k + ${2 * j}u < P.K);`);
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let base = (P.bOff + z * P.bBatch + n * P.K + k) >> 1u;
  ${w.join("\n  ")}
  return V8(vec4<f32>(w0, w1), vec4<f32>(w2, w3));
}`;
  }
  if (b === "i8") {
    const load = vec
      ? "let w = B[(n * P.K + k) >> 3u];\n  let w0 = i32(w.x);\n  let w1 = i32(w.y);"
      : "let base = (n * P.K + k) >> 2u;\n  let w0 = i32(B[base]);\n  let w1 = i32(B[base + 1u]);";
    const x = (w) => range(4).map((j) => `f32(extractBits(${w}, ${8 * j}u, 8u))`).join(", ");
    return `
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  ${load}
  return V8(vec4<f32>(${x("w0")}), vec4<f32>(${x("w1")}));
}`;
  }
  if (b === "w4") {
    const lv = (j) => `clamp(round(CB[(codes >> ${4 * j}u) & 15u] * s), -127.0, 127.0)`;
    return `
fn fp8e4m3(b: u32) -> f32 {
  let e = (b >> 3u) & 15u;
  let m = f32(b & 7u);
  var v: f32;
  if (e == 0u) { v = m * 0.001953125; } else { v = (1.0 + m * 0.125) * exp2(f32(e) - 7.0); }
  if ((b & 128u) != 0u) { v = -v; }
  return v;
}
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let codes = B[(n * P.K + k) >> 3u];
  let gi = (n * P.K + k) >> 4u;
  let s = fp8e4m3((SR[gi >> 2u] >> ((gi & 3u) * 8u)) & 255u);
  return V8(vec4<f32>(${range(4).map(lv).join(", ")}), vec4<f32>(${range(4).map((j) => lv(j + 4)).join(", ")}));
}`;
  }
  throw new Error(`unknown B format ${b}`);
}

// Epilogue statements on r for output (m, n). scaled: per-row weight scale RS[n].
export function epilogue({ scaled = false, bias = false, add = false, act = "none" }) {
  let epi = "";
  if (scaled) epi += " r = r * RS[n];";
  if (bias) epi += " r = r + BIAS[n];";
  if (add) epi += " r = r + ADD[m * P.N + n];";
  if (act === "relu") epi += " r = max(r, 0.0);";
  if (act === "leaky") epi += " r = select(r * P.slope, r, r >= 0.0);";
  if (act === "prelu") epi += " r = select(r * SL[n], r, r >= 0.0);";
  if (act === "silu") epi += " r = r / (1.0 + exp(-r));";
  // (tanh(r) + 1) / 2 written as sigmoid(2r): saturates cleanly where some tanh implementations
  // overflow to NaN
  if (act === "tanh01") epi += " r = 1.0 / (1.0 + exp(-2.0 * r));";
  return epi;
}

// a: "rows" | { conv: true, vec, pad, up, pro }
// vecA / vecB: vectorized global loads; the caller guarantees K % 8 == 0 and 4-aligned offsets
// and strides (for the weight formats: K % 8 == 0).
// add: an extra [M, N] input added before the activation, ADD[m*N + n].
export function matmulShader({ a = "rows", b = "f32", bias = false, act = "none", resid = false, gate = false, R = 8, vecA = false, vecB = false, add = false, split = false, phases = false }) {
  const key = JSON.stringify([a, b, bias, act, resid, gate, R, vecA, vecB, add, split, phases]);
  if (cache.has(key)) return cache.get(key);
  const conv = typeof a === "object";
  const vecIn = conv ? a.vec : vecA;
  if (split) bias = add = gate = resid = false;

  const scaled = !split && (b === "i8" || b === "w4");
  const bindings = [];
  const bind = (decl) => bindings.push(`@group(0) @binding(${bindings.length + 1}) ${decl};`);
  bind(`var<storage, read> A: array<${vecIn ? "vec4<f32>" : "f32"}>`);
  bind(`var<storage, read> B: array<${weightArrayType(b, vecB)}>`);
  bind("var<storage, read_write> C: array<f32>");
  if (scaled) bind("var<storage, read> RS: array<f32>");
  if (b === "w4") { bind("var<storage, read> SR: array<u32>"); bind("var<storage, read> CB: array<f32>"); }
  if (bias) bind("var<storage, read> BIAS: array<f32>");
  if (gate) bind("var<storage, read> G: array<f32>");
  if (add) bind("var<storage, read> ADD: array<f32>");
  if (act === "prelu" && !split) bind("var<storage, read> SL: array<f32>");
  if (conv && a.pro) bind(`var<storage, read> PRO: array<${a.vec ? "vec4<f32>" : "f32"}>`);

  const epi = split ? "" : epilogue({ scaled, bias, add, act });
  const store = resid ? (gate ? "C[ci] = C[ci] + G[P.gOff + n] * r;" : "C[ci] = C[ci] + r;") : "C[ci] = r;";
  // split: raw partial sums at [z][m][n]
  const cIndex = split ? "z * P.cBatch + m * P.N + n" : "P.cOff + z * P.cBatch + m * P.ldc + n";
  const scale = split ? "" : " * P.alpha";

  const T = 16 * R; // tile edge
  const TK = R === 8 ? 16 : 32;
  const V = T / 4; // vec4 per k-row of a tile
  const G = R / 4; // 64-wide row/column groups per thread
  const perRow = TK / 8; // threads per staged row

  // staging: 8 k values of row lr -> As[(lk + j) * V + (lr >> 2)][lr & 3]
  const stage = range(8).map((j) => {
    const src = j < 4 ? `lo.${"xyzw"[j]}` : `hi.${"xyzw"[j - 4]}`;
    return `As[(lk + ${j}u) * ${V}u + sIdx][sLane] = av.${src};\n      Bs[(lk + ${j}u) * ${V}u + sIdx][sLane] = bv.${src};`;
  });

  const fma = [];
  for (let g = 0; g < G; g++) fma.push(`let a${g} = As[kk * ${V}u + ${g * 16}u + tm];`);
  for (let h = 0; h < G; h++) fma.push(`let b${h} = Bs[kk * ${V}u + ${h * 16}u + tn];`);
  for (let g = 0; g < G; g++) {
    ["x", "y", "z", "w"].forEach((c, i) => {
      for (let h = 0; h < G; h++) fma.push(`acc${(g * 4 + i) * G + h} += a${g}.${c} * b${h};`);
    });
  }

  // epilogue: fully unrolled over the thread's R x R outputs
  const out = [];
  for (let g = 0; g < G; g++) {
    for (let i = 0; i < 4; i++) {
      out.push(`{ let m = m0 + ${g * 64 + i}u + tm * 4u;\n    if (m < P.M) {`);
      for (let h = 0; h < G; h++) {
        for (let j = 0; j < 4; j++) {
          out.push(`      { let n = n0 + ${h * 64 + j}u + tn * 4u; if (n < P.N) { let ci = ${cIndex}; var r = acc${(g * 4 + i) * G + h}.${"xyzw"[j]}${scale};${epi} ${store} } }`);
        }
      }
      out.push("    } }");
    }
  }

  const code = /* wgsl */ `
struct Params {
  M: u32, N: u32, K: u32, alpha: f32,
  lda: u32, aBatch: u32, aOff: u32, ldb: u32,
  bBatch: u32, bDiv: u32, bOff: u32, ldc: u32,
  cBatch: u32, cOff: u32, gOff: u32, cin: u32,
  ih: u32, iw: u32, oh: u32, ow: u32,
  kw: u32, stride: u32, pad: u32, slope: f32,
  proOff: u32, proStride: u32, kChunk: u32, phases: u32,
};
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${bindings.join("\n")}
${conv ? loaderConv(a) : loaderRows(vecA)}
${loaderB(b, vecB)}

const TK = ${TK}u;
var<workgroup> As: array<vec4<f32>, ${V * TK}>;  // [TK][T/4]
var<workgroup> Bs: array<vec4<f32>, ${V * TK}>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let z = wg.z;
  ${split ? `let kBegin = z * P.kChunk;
  let kEnd = min(P.K, kBegin + P.kChunk);
  let zl = 0u; // the split index isn't a batch index` : phases ? `// the taps of the phases this workgroup's columns belong to
  let ph0 = (wg.x * ${T}u) / P.phases;
  let ph1 = min((wg.x * ${T}u + ${T - 1}u) / P.phases, 3u);
  let kBegin = min(ph0, 2u) * P.cin;
  let kEnd = select(3u, 4u, ph0 == 0u || (ph0 <= 2u && ph1 >= 2u)) * P.cin;
  let zl = z;` : `let kBegin = 0u;
  let kEnd = P.K;
  let zl = z;`}
  let m0 = wg.y * ${T}u;
  let n0 = wg.x * ${T}u;
  let tm = t & 15u;
  let tn = t >> 4u;
  let lr = t / ${perRow}u;
  let lk = (t % ${perRow}u) * 8u;
  let sIdx = lr >> 2u;
  let sLane = lr & 3u;
  ${range(R * G).map((i) => `var acc${i} = vec4<f32>();`).join("\n  ")}
  var av = loadA8(zl, m0 + lr, kBegin + lk);
  var bv = loadB8(zl, n0 + lr, kBegin + lk);

  for (var k0 = kBegin; k0 < kEnd; k0 += TK) {
    ${stage.join("\n    ")}
    workgroupBarrier();
    if (k0 + TK < kEnd) {
      av = loadA8(zl, m0 + lr, k0 + TK + lk);
      bv = loadB8(zl, n0 + lr, k0 + TK + lk);
    }
    for (var kk = 0u; kk < TK; kk++) {
      ${fma.join("\n      ")}
    }
    workgroupBarrier();
  }

  ${out.join("\n  ")}
}`;
  cache.set(key, code);
  return code;
}
