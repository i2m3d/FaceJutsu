// WGSL kernels other than the GEMM. Every kernel binds its parameters as a uniform at binding 0.
// Activations are NHWC f32; frames are RGBA8 packed one pixel per u32, rows `stride` pixels apart.

const cache = new Map();
const memo = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn());
  return cache.get(key);
};

import { epilogue, loaderB, weightArrayType } from "./gemm.js";

export { matmulShader } from "./gemm.js";

// Split-K reduction: C[m][n] = epilogue(alpha * sum_s PART[s][m][n]), stored at
// C[cOff + m*ldc + n] (resid: added to it). Same epilogue options as the GEMM.
export const splitReduceShader = ({ scaled = false, bias = false, add = false, act = "none", resid = false }) =>
  memo(`splitred${scaled}${bias}${add}${act}${resid}`, () => {
    const bindings = [];
    const bind = (decl) => bindings.push(`@group(0) @binding(${bindings.length + 1}) ${decl};`);
    bind("var<storage, read> PART: array<f32>");
    bind("var<storage, read_write> C: array<f32>");
    if (scaled) bind("var<storage, read> RS: array<f32>");
    if (bias) bind("var<storage, read> BIAS: array<f32>");
    if (add) bind("var<storage, read> ADD: array<f32>");
    if (act === "prelu") bind("var<storage, read> SL: array<f32>");
    return /* wgsl */ `
struct Params { M: u32, N: u32, S: u32, alpha: f32, nx: u32, slope: f32, ldc: u32, cOff: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${bindings.join("\n")}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  let MN = P.M * P.N;
  if (i >= MN) { return; }
  let m = i / P.N;
  let n = i % P.N;
  var r = 0.0;
  for (var s = 0u; s < P.S; s++) { r += PART[s * MN + i]; }
  r = r * P.alpha;${epilogue({ scaled, bias, add, act })}
  let ci = P.cOff + m * P.ldc + n;
  ${resid ? "C[ci] = C[ci] + r;" : "C[ci] = r;"}
}`;
  });

// y = x @ H per group of 256 features, H = kron(H4,H4,H4,H4)/16: one 4-point butterfly per digit.
// Operates on S[0, cols) in workgroup memory with 256 threads (cols a multiple of 256, <= 2048).
const HADAMARD = /* wgsl */ `
fn hadamardRows(t: u32, cols: u32) {
  let quads = cols / 4u;
  var stride = 1u;
  for (var d = 0u; d < 4u; d++) {
    for (var q = t; q < quads; q += 256u) {
      let g = q / 64u;
      let w = q % 64u;
      let i0 = g * 256u + (w / stride) * stride * 4u + w % stride;
      let a = S[i0]; let b = S[i0 + stride]; let c = S[i0 + 2u * stride]; let e = S[i0 + 3u * stride];
      // regular H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
      S[i0] = a + b + c - e;
      S[i0 + stride] = a + b - c + e;
      S[i0 + 2u * stride] = a - b + c + e;
      S[i0 + 3u * stride] = -a + b + c + e;
    }
    workgroupBarrier();
    stride *= 4u;
  }
}`;

// ConvRot input rotation: Y = X @ H per contiguous group of 256 features.
export const hadamardShader = () => memo("had", () => /* wgsl */ `
struct Params { groups: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> s: array<f32, 256>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.y * P.nx + wg.x;
  if (g >= P.groups) { return; }
  let base = g * 256u;
  for (var j = 0u; j < 4u; j++) { s[t + 64u * j] = X[base + t + 64u * j]; }
  workgroupBarrier();
  var stride = 1u;
  for (var d = 0u; d < 4u; d++) {
    let lo = t % stride;
    let i0 = (t / stride) * stride * 4u + lo;
    let a = s[i0]; let b = s[i0 + stride]; let c = s[i0 + 2u * stride]; let e = s[i0 + 3u * stride];
    workgroupBarrier();
    s[i0] = a + b + c - e;
    s[i0 + stride] = a + b - c + e;
    s[i0 + 2u * stride] = a - b + c + e;
    s[i0 + 3u * stride] = -a + b + c + e;
    workgroupBarrier();
    stride *= 4u;
  }
  for (var j = 0u; j < 4u; j++) { Y[base + t + 64u * j] = s[t + 64u * j] * 0.0625; }
}`);

// Instance norm statistics + AdaIN, folded into the per-(image, channel) affine the next step
// applies: for X [B, HW, C] and a style row [scale(C), shift(C)] per image,
//   S = scale / sqrt(var + eps),  T = shift - mean * S        (PRO = [S (B*C), T (B*C)])
// Two passes over the pixels (mean, then squared deviations), as the ONNX graph computes it.
// One workgroup per (64 channels, image): 16 vec4 channel columns x 16 pixel lanes.
export const adainShader = () => memo("adain", () => /* wgsl */ `
struct Params { HW: u32, C: u32, B: u32, styleStride: u32, eps: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> STY: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> PRO: array<vec4<f32>>;
var<workgroup> red: array<vec4<f32>, 256>;
// sum over the 16 pixel lanes of each channel column
fn reduce(t: u32, v: vec4<f32>) -> vec4<f32> {
  red[t] = v;
  workgroupBarrier();
  for (var s = 128u; s >= 16u; s >>= 1u) {
    if (t < s) { red[t] = red[t] + red[t + s]; }
    workgroupBarrier();
  }
  let r = red[t % 16u];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let b = wg.y;
  let lane = t / 16u;                // pixel lane
  let c4 = wg.x * 16u + t % 16u;     // vec4 channel index
  let C4 = P.C / 4u;
  let base = b * P.HW * C4 + c4;
  var s = vec4<f32>(0.0);
  for (var p = lane; p < P.HW; p += 16u) { s += X[base + p * C4]; }
  let mean = reduce(t, s) / f32(P.HW);
  var v = vec4<f32>(0.0);
  for (var p = lane; p < P.HW; p += 16u) { let d = X[base + p * C4] - mean; v += d * d; }
  let rstd = 1.0 / sqrt(reduce(t, v) / f32(P.HW) + P.eps);
  if (lane == 0u) {
    let st = b * P.styleStride / 4u;
    let sc = STY[st + c4] * rstd;
    PRO[b * C4 + c4] = sc;
    PRO[P.B * C4 + b * C4 + c4] = STY[st + C4 + c4] - mean * sc;
  }
}`);

// The same statistics for large images, in two steps: normPartsShader gives every (image, chunk of
// pixels, 64 channels) workgroup its chunk's mean and sum of squared deviations (two passes over
// the chunk), normCombineShader merges the chunks (Chan et al.'s pairwise update) and folds in the
// style row as adainShader does. PART: [B][chunks][mean C, M2 C].
export const normPartsShader = () => memo("normparts", () => /* wgsl */ `
struct Params { HW: u32, C: u32, chunk: u32, chunks: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> PART: array<vec4<f32>>;
var<workgroup> red: array<vec4<f32>, 256>;
fn reduce(t: u32, v: vec4<f32>) -> vec4<f32> {
  red[t] = v;
  workgroupBarrier();
  for (var s = 128u; s >= 16u; s >>= 1u) {
    if (t < s) { red[t] = red[t] + red[t + s]; }
    workgroupBarrier();
  }
  let r = red[t % 16u];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let b = wg.z;
  let k = wg.y;
  let lane = t / 16u;
  let c4 = wg.x * 16u + t % 16u;
  let C4 = P.C / 4u;
  let p0 = k * P.chunk;
  let p1 = min(p0 + P.chunk, P.HW);
  let base = b * P.HW * C4 + c4;
  var s = vec4<f32>(0.0);
  for (var p = p0 + lane; p < p1; p += 16u) { s += X[base + p * C4]; }
  let mean = reduce(t, s) / f32(p1 - p0);
  var v = vec4<f32>(0.0);
  for (var p = p0 + lane; p < p1; p += 16u) { let d = X[base + p * C4] - mean; v += d * d; }
  let m2 = reduce(t, v);
  if (lane == 0u) {
    let o = (b * P.chunks + k) * 2u * C4 + c4;
    PART[o] = mean;
    PART[o + C4] = m2;
  }
}`);

export const normCombineShader = () => memo("normcombine", () => /* wgsl */ `
struct Params { HW: u32, C: u32, B: u32, chunk: u32, chunks: u32, styleStride: u32, eps: f32, p0: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> PART: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> STY: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> PRO: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let C4 = P.C / 4u;
  if (gid.x >= P.B * C4) { return; }
  let b = gid.x / C4;
  let c4 = gid.x % C4;
  var n = 0.0;
  var mean = vec4<f32>(0.0);
  var m2 = vec4<f32>(0.0);
  for (var k = 0u; k < P.chunks; k++) {
    let nk = f32(min(P.chunk, P.HW - k * P.chunk));
    let o = (b * P.chunks + k) * 2u * C4 + c4;
    let tot = n + nk;
    let d = PART[o] - mean;
    mean += d * (nk / tot);
    m2 += PART[o + C4] + d * d * (n * nk / tot);
    n = tot;
  }
  let rstd = 1.0 / sqrt(m2 / f32(P.HW) + P.eps);
  let st = b * P.styleStride / 4u;
  let sc = STY[st + c4] * rstd;
  PRO[b * C4 + c4] = sc;
  PRO[P.B * C4 + b * C4 + c4] = STY[st + C4 + c4] - mean * sc;
}`);

// Per-(image, channel) affine on X [B, HW, C] with PRO = [S, T] (proStride = C, or 0 when every
// image shares one row), one workgroup of 256 threads per pixel:
//   mode "relu":   Y = max(X*S + T, 0)
//   mode "plain":  Y = X*S + T
//   mode "resid":  Y += X*S + T          (residual add, in place on Y)
//   mode "scale":  Y = X*S, S = PRO[proOff + b*proStride + c]   (a modulated conv's style)
// rotate: Y is written ConvRot-rotated (input of a quantized conv; C % 256 == 0, C <= 2048).
export const affineShader = (mode, rotate) => memo(`aff${mode}${rotate}`, () => /* wgsl */ `
struct Params { rows: u32, C: u32, HW: u32, proStride: u32, proOff: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> PRO: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
${rotate ? "var<workgroup> S: array<f32, 2048>;\n" + HADAMARD : ""}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let b = row / P.HW;
  let base = row * P.C;
  let pb = b * P.proStride;
  for (var c = t; c < P.C; c += 256u) {
    var v = ${mode === "scale" ? "X[base + c] * PRO[P.proOff + pb + c]" : "X[base + c] * PRO[pb + c] + PRO[P.proOff + pb + c]"};
    ${mode === "relu" ? "v = max(v, 0.0);" : ""}
    ${rotate ? "S[c] = v;" : mode === "resid" ? "Y[base + c] = Y[base + c] + v;" : "Y[base + c] = v;"}
  }
  ${rotate ? `workgroupBarrier();
  hadamardRows(t, P.C);
  for (var c = t; c < P.C; c += 256u) { Y[base + c] = S[c] * 0.0625; }` : ""}
}`);

// 2x2 stride-2 pooling on NHWC [B, H, W, C] (H, W even): "max" or "avg".
export const pool2Shader = (op) => memo("pool" + op, () => /* wgsl */ `
struct Params { n: u32, C: u32, OW: u32, OH: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let c = i % P.C;
  let ox = (i / P.C) % P.OW;
  let r = i / (P.C * P.OW);       // b * OH + oy
  let W = P.OW * 2u;
  let top = ((r * 2u) * W + ox * 2u) * P.C + c;
  let bot = top + W * P.C;
  let a = X[top]; let b = X[top + P.C]; let d = X[bot]; let e = X[bot + P.C];
  Y[i] = ${op === "max" ? "max(max(a, b), max(d, e))" : "(a + b + d + e) * 0.25"};
}`);

// Y[b, y, x, c] += X[b, y/2, x/2, c] (FPN top-down path: nearest 2x upsample, then add)
export const upAddShader = () => memo("upadd", () => /* wgsl */ `
struct Params { n: u32, C: u32, W: u32, H: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let c = i % P.C;
  let x = (i / P.C) % P.W;
  let r = i / (P.C * P.W);
  let y = r % P.H;
  let b = r / P.H;
  Y[i] = Y[i] + X[((b * (P.H / 2u) + y / 2u) * (P.W / 2u) + x / 2u) * P.C + c];
}`);

// Affine resampling of an RGBA8 frame into an NHWC RGB f32 tensor, with pixel centers at integer
// coordinates (as cv2.warpAffine): out(x, y) = frame(M @ (x, y, 1)) * mul + add. When the map
// shrinks, n x n bilinear samples are averaged per output pixel (anti-aliasing). Pixels at or
// beyond (vw, vh) get `fill` (the detector letterbox). border: edge pixels replicated, or zero
// (zero: 1). k > 1 writes the pixelshuffle layout: output pixel (y, x) goes to sub-image
// (y % k) * k + x % k at (y / k, x / k). OUT is written from element `off` on.
export const resampleShader = () => memo("resample", () => /* wgsl */ `
struct Params {
  w: u32, h: u32, stride: u32, ow: u32,
  oh: u32, vw: u32, vh: u32, k: u32,
  m0: f32, m1: f32, m2: f32, m3: f32,
  m4: f32, m5: f32, n: u32, zero: u32,
  mul: f32, add: f32, fill: f32, off: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> FRAME: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
fn px(x: i32, y: i32) -> vec3<f32> {
  if (P.zero == 1u && (x < 0 || y < 0 || x >= i32(P.w) || y >= i32(P.h))) { return vec3<f32>(0.0); }
  let cx = u32(clamp(x, 0, i32(P.w) - 1));
  let cy = u32(clamp(y, 0, i32(P.h) - 1));
  return unpack4x8unorm(FRAME[cy * P.stride + cx]).xyz * 255.0;
}
fn bilinear(sx: f32, sy: f32) -> vec3<f32> {
  let fx = floor(sx);
  let fy = floor(sy);
  let x0 = i32(fx);
  let y0 = i32(fy);
  let wx = sx - fx;
  let wy = sy - fy;
  return mix(mix(px(x0, y0), px(x0 + 1, y0), wx), mix(px(x0, y0 + 1), px(x0 + 1, y0 + 1), wx), wy);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  if (ox >= P.ow || oy >= P.oh) { return; }
  var v = vec3<f32>(P.fill);
  if (ox < P.vw && oy < P.vh) {
    var acc = vec3<f32>(0.0);
    let n = f32(P.n);
    for (var j = 0u; j < P.n; j++) {
      for (var i = 0u; i < P.n; i++) {
        let x = f32(ox) + (f32(i) + 0.5) / n - 0.5;
        let y = f32(oy) + (f32(j) + 0.5) / n - 0.5;
        acc += bilinear(P.m0 * x + P.m1 * y + P.m2, P.m3 * x + P.m4 * y + P.m5);
      }
    }
    v = acc / (n * n) * P.mul + P.add;
  }
  let sw = P.ow / P.k;
  let sh = P.oh / P.k;
  let sub = (oy % P.k) * P.k + ox % P.k;
  let o = P.off + ((sub * sh + oy / P.k) * sw + ox / P.k) * 3u;
  OUT[o] = v.x;
  OUT[o + 1u] = v.y;
  OUT[o + 2u] = v.z;
}`);

// Blends a swapped face back into the RGBA8 frame, in place. For every frame pixel in the box
// [x1, x2) x [y1, y2): its crop position c = M @ (x, y, 1) samples the swapped crop (bilinear,
// edges replicated; pixelshuffle layout with k*k sub-images of `sub` pixels) and the mask
// (bilinear on the outer product of the 1D profile MASK[S], zero outside) times alpha, and
//   frame = frame * (1 - a) + crop * a,
// or with add, the crop a change to the frame: frame = frame + crop * a.
export const pasteShader = (add = false) => memo(`paste${add}`, () => /* wgsl */ `
struct Params {
  stride: u32, x1: u32, y1: u32, x2: u32,
  y2: u32, S: u32, k: u32, sub: u32,
  m0: f32, m1: f32, m2: f32, m3: f32,
  m4: f32, m5: f32, alpha: f32, p0: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> FRAME: array<u32>;
@group(0) @binding(2) var<storage, read> SW: array<f32>;
@group(0) @binding(3) var<storage, read> MASK: array<f32>;
fn cropPx(x: i32, y: i32) -> vec3<f32> {
  let cx = u32(clamp(x, 0, i32(P.S) - 1));
  let cy = u32(clamp(y, 0, i32(P.S) - 1));
  let s = (cy % P.k) * P.k + cx % P.k;
  let o = ((s * P.sub + cy / P.k) * P.sub + cx / P.k) * 3u;
  return ${add ? "vec3<f32>(SW[o], SW[o + 1u], SW[o + 2u])" : "clamp(vec3<f32>(SW[o], SW[o + 1u], SW[o + 2u]), vec3<f32>(0.0), vec3<f32>(1.0))"};
}
fn prof(i: i32) -> f32 {
  if (i < 0 || i >= i32(P.S)) { return 0.0; }
  return MASK[u32(i)];
}
fn lin(c: f32) -> f32 {
  let f = floor(c);
  let i = i32(f);
  return mix(prof(i), prof(i + 1), c - f);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = P.x1 + gid.x;
  let y = P.y1 + gid.y;
  if (x >= P.x2 || y >= P.y2) { return; }
  let fx = f32(x);
  let fy = f32(y);
  let cx = P.m0 * fx + P.m1 * fy + P.m2;
  let cy = P.m3 * fx + P.m4 * fy + P.m5;
  let a = clamp(lin(cx) * lin(cy), 0.0, 1.0) * P.alpha;
  if (a <= 0.0) { return; }
  let ix = floor(cx);
  let iy = floor(cy);
  let x0 = i32(ix);
  let y0 = i32(iy);
  let sw = mix(mix(cropPx(x0, y0), cropPx(x0 + 1, y0), cx - ix), mix(cropPx(x0, y0 + 1), cropPx(x0 + 1, y0 + 1), cx - ix), cy - iy);
  let i = y * P.stride + x;
  let f = unpack4x8unorm(FRAME[i]).xyz;
  FRAME[i] = pack4x8unorm(vec4<f32>(${add ? "f + sw * a" : "mix(f, sw, a)"}, 1.0));
}`);

// Puts back what covers a face, the last step of a frame: for every frame pixel in the box
// [x1, x2) x [y1, y2), a = the face's occlusion mask (an N x N image at OCC[oOff..], sampled
// through its map O @ (x, y, 1), bilinear, 1 outside its crop) and
//   frame = orig + (frame - orig) * a,
// orig the frame before any swap (same stride).
export const unoccludeShader = () => memo("unocclude", () => /* wgsl */ `
struct Params {
  stride: u32, x1: u32, y1: u32, x2: u32,
  y2: u32, N: u32, oOff: u32, p0: u32,
  o0: f32, o1: f32, o2: f32, o3: f32,
  o4: f32, o5: f32, p1: u32, p2: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> FRAME: array<u32>;
@group(0) @binding(2) var<storage, read> ORIG: array<u32>;
@group(0) @binding(3) var<storage, read> OCC: array<f32>;
fn occPx(x: i32, y: i32) -> f32 {
  if (x < 0 || y < 0 || x >= i32(P.N) || y >= i32(P.N)) { return 1.0; }
  return OCC[P.oOff + u32(y) * P.N + u32(x)];
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = P.x1 + gid.x;
  let y = P.y1 + gid.y;
  if (x >= P.x2 || y >= P.y2) { return; }
  let fx = f32(x);
  let fy = f32(y);
  let ox = P.o0 * fx + P.o1 * fy + P.o2;
  let oy = P.o3 * fx + P.o4 * fy + P.o5;
  let ix = floor(ox);
  let iy = floor(oy);
  let x0 = i32(ix);
  let y0 = i32(iy);
  let a = mix(mix(occPx(x0, y0), occPx(x0 + 1, y0), ox - ix), mix(occPx(x0, y0 + 1), occPx(x0 + 1, y0 + 1), ox - ix), oy - iy);
  if (a >= 1.0) { return; }
  let i = y * P.stride + x;
  let o = unpack4x8unorm(ORIG[i]).xyz;
  let f = unpack4x8unorm(FRAME[i]).xyz;
  FRAME[i] = pack4x8unorm(vec4<f32>(mix(o, f, a), 1.0));
}`);

// ----------------------------------------------------------------------------- Winograd F(2x2, 3x3)
//
// A stride-1, pad-1 3x3 convolution as 16 GEMMs: every 2x2 output tile reads a 4x4 input patch d,
// the input transform writes V = B^T d B, the GEMMs multiply each of the 16 positions by the
// transformed filters U = G g G^T ([16][Cout][Cin], prepared on load), and the output transform
// turns M = U V back into the 2x2 tile, A^T M A. 2.25x fewer multiplies than the direct conv.

// Input transform of tiles t0.. of a batch of NHWC images (numbered image by image, row by row)
// into V [16][T][C] (T = tiles of this chunk), with the direct conv loader's options: pad "zeros" |
// "reflect", up "" | "nearest" | "bilinear" (input upsampled 2x first), pro "" | "affine" |
// "affine_relu" (per (image, channel) x*S + T on the input; padding stays zero) | "scale" (x*S,
// S at proOff + b*proStride). One thread per (tile, 4 channels).
export const winogradInShader = ({ pad = "zeros", up = "", pro = "" }) => memo(`wgin${pad}${up}${pro}`, () => {
  const LH = up ? "2 * i32(P.IH)" : "i32(P.IH)";
  const LW = up ? "2 * i32(P.IW)" : "i32(P.IW)";
  const bounds = pad === "reflect"
    ? `y = refl(y, ${LH}); x = refl(x, ${LW});`
    : `if (y < 0 || x < 0 || y >= ${LH} || x >= ${LW}) { return vec4<f32>(0.0); }`;
  let fetch;
  if (up === "bilinear") {
    fetch = `
  let sy = max((f32(y) + 0.5) * 0.5 - 0.5, 0.0);
  let sx = max((f32(x) + 0.5) * 0.5 - 0.5, 0.0);
  let y0 = u32(sy);
  let x0 = u32(sx);
  let y1 = min(y0 + 1u, P.IH - 1u);
  let x1 = min(x0 + 1u, P.IW - 1u);
  let r0 = (b * P.IH + y0) * P.IW;
  let r1 = (b * P.IH + y1) * P.IW;
  let C4 = P.C / 4u;
  var v = mix(mix(X[(r0 + x0) * C4 + c4], X[(r0 + x1) * C4 + c4], sx - f32(x0)),
              mix(X[(r1 + x0) * C4 + c4], X[(r1 + x1) * C4 + c4], sx - f32(x0)), sy - f32(y0));`;
  } else {
    const yy = up === "nearest" ? "u32(y) / 2u" : "u32(y)";
    const xx = up === "nearest" ? "u32(x) / 2u" : "u32(x)";
    fetch = `\n  var v = X[((b * P.IH + ${yy}) * P.IW + ${xx}) * (P.C / 4u) + c4];`;
  }
  const prologue = pro === "scale" ? "\n  v = v * PRO[(P.proOff + b * P.proStride) / 4u + c4];"
    : pro ? `\n  let si = (b * P.proStride) / 4u + c4;\n  v = v * PRO[si] + PRO[P.proOff / 4u + si];${pro === "affine_relu" ? "\n  v = max(v, vec4<f32>(0.0));" : ""}`
    : "";
  const d = (i, j) => `d${i}${j}`;
  const loads = [];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) loads.push(`let ${d(i, j)} = px(b, y0 + ${i}, x0 + ${j}, c4);`);
  // t = B^T d (rows), then V = t B (columns): B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
  const rows = (c) => [`${c(0)} - ${c(2)}`, `${c(1)} + ${c(2)}`, `${c(2)} - ${c(1)}`, `${c(1)} - ${c(3)}`];
  const t = [];
  for (let j = 0; j < 4; j++) rows((i) => d(i, j)).forEach((e, i) => t.push(`let t${i}${j} = ${e};`));
  const out = [];
  for (let i = 0; i < 4; i++) rows((j) => `t${i}${j}`).forEach((e, j) => out.push(`V[(${i * 4 + j}u * P.T + tl) * C4 + c4] = ${e};`));
  return /* wgsl */ `
struct Params { IH: u32, IW: u32, C: u32, th: u32, tw: u32, T: u32, t0: u32, nx: u32, proOff: u32, proStride: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> V: array<vec4<f32>>;
${pro ? "@group(0) @binding(3) var<storage, read> PRO: array<vec4<f32>>;" : ""}
fn refl(i: i32, n: i32) -> i32 { return select(select(i, 2 * n - 2 - i, i >= n), -i, i < 0); }
fn px(b: u32, yy: i32, xx: i32, c4: u32) -> vec4<f32> {
  var y = yy;
  var x = xx;
  ${bounds}${fetch}${prologue}
  return v;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let C4 = P.C / 4u;
  let idx = gid.y * P.nx * 64u + gid.x;
  if (idx >= P.T * C4) { return; }
  let c4 = idx % C4;
  let tl = idx / C4;
  let tg = P.t0 + tl;
  let tx = tg % P.tw;
  let r = tg / P.tw;
  let b = r / P.th;
  let y0 = 2 * i32(r % P.th) - 1;
  let x0 = 2 * i32(tx) - 1;
  ${loads.join("\n  ")}
  ${t.join("\n  ")}
  ${out.join("\n  ")}
}`;
});

// Filter preparation: a 3x3 conv weight in any stored format ([N, 9*C], OHWI) -> f16, either
//   out "winograd": U [16][N][C] = G g G^T, G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//   out "ohwi":     the dense filter [N][9][C] (the WebNN backend's constants)
// A ConvRot rotation (rotated: each tap's input channels were multiplied by H per group of 256) is
// undone first, with the same butterflies as the activation rotation. One workgroup per (gs input
// channels, output channel n); each thread decodes 8 channels of all 9 taps through the GEMM's
// weight loader.
export const filterPrepShader = ({ b, vec, rotated, gs, out = "winograd" }) => memo(`fprep${b}${vec}${rotated}${gs}${out}`, () => {
  const threads = gs / 8;
  const pack8 = (v) => `vec4<u32>(pack2x16float(vec2<f32>(${v(0)}, ${v(1)})), pack2x16float(vec2<f32>(${v(2)}, ${v(3)})),
      pack2x16float(vec2<f32>(${v(4)}, ${v(5)})), pack2x16float(vec2<f32>(${v(6)}, ${v(7)})))`;
  const write = out === "ohwi" ? `
  for (var tap = 0u; tap < 9u; tap++) {
    let o = tap * ${gs}u + t * 8u;
    U[((n * 9u + tap) * P.C + c0) / 8u] = ${pack8((j) => `S[o + ${j}u] * norm`)};
  }` : `
  var u: array<array<f32, 8>, 16>;
  for (var j = 0u; j < 8u; j++) {
    let c = t * 8u + j;
    var g: array<f32, 9>;
    for (var tap = 0u; tap < 9u; tap++) { g[tap] = S[tap * ${gs}u + c] * norm; }
    // t = G g (4x3), then U = t G^T (4x4)
    for (var y = 0u; y < 4u; y++) {
      var r: vec3<f32>;
      for (var x = 0u; x < 3u; x++) {
        let g0 = g[x]; let g1 = g[3u + x]; let g2 = g[6u + x];
        r[x] = select(select(select(g2, 0.5 * (g0 - g1 + g2), y == 2u), 0.5 * (g0 + g1 + g2), y == 1u), g0, y == 0u);
      }
      u[y * 4u][j] = r.x;
      u[y * 4u + 1u][j] = 0.5 * (r.x + r.y + r.z);
      u[y * 4u + 2u][j] = 0.5 * (r.x - r.y + r.z);
      u[y * 4u + 3u][j] = r.z;
    }
  }
  for (var p = 0u; p < 16u; p++) {
    U[((p * P.N + n) * P.C + c0) / 8u] = ${pack8((j) => `u[p][${j}]`)};
  }`;
  const scaled = b === "i8" || b === "w4";
  const bindings = [];
  const bind = (decl) => bindings.push(`@group(0) @binding(${bindings.length + 1}) ${decl};`);
  bind(`var<storage, read> B: array<${weightArrayType(b, vec)}>`);
  if (scaled) bind("var<storage, read> RS: array<f32>");
  if (b === "w4") { bind("var<storage, read> SR: array<u32>"); bind("var<storage, read> CB: array<f32>"); }
  bind("var<storage, read_write> U: array<vec4<u32>>");
  const lanes = ["lo.x", "lo.y", "lo.z", "lo.w", "hi.x", "hi.y", "hi.z", "hi.w"];
  return /* wgsl */ `
struct Params { N: u32, K: u32, ldb: u32, bOff: u32, bBatch: u32, bDiv: u32, C: u32, p0: u32 };
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${bindings.join("\n")}
${loaderB(b, vec)}
var<workgroup> S: array<f32, ${9 * gs}>;
@compute @workgroup_size(${threads})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = wg.y;
  let c0 = wg.x * ${gs}u + t * 8u;
  let s = ${scaled ? "RS[n]" : "1.0"};
  for (var tap = 0u; tap < 9u; tap++) {
    let v = loadB8(0u, n, tap * P.C + c0);
    let o = tap * ${gs}u + t * 8u;
    ${lanes.map((l, j) => `S[o + ${j}u] = v.${l} * s;`).join(" ")}
  }
  workgroupBarrier();
  ${rotated ? `// undo ConvRot: x @ H per tap, H = kron(H4,H4,H4,H4)/16
  for (var stride = 1u; stride < 256u; stride *= 4u) {
    for (var q = t; q < 9u * 64u; q += ${threads}u) {
      let i0 = (q / 64u) * 256u + ((q % 64u) / stride) * stride * 4u + (q % 64u) % stride;
      let a = S[i0]; let b = S[i0 + stride]; let c = S[i0 + 2u * stride]; let e = S[i0 + 3u * stride];
      S[i0] = a + b + c - e;
      S[i0 + stride] = a + b - c + e;
      S[i0 + 2u * stride] = a - b + c + e;
      S[i0 + 3u * stride] = -a + b + c + e;
    }
    workgroupBarrier();
  }` : ""}
  let norm = ${rotated ? "0.0625" : "1.0"};${write}
}`;
});

// A matmul weight [N, K] in any stored format -> dense f16 [N][K] (the WebNN backend's
// constants), a ConvRot rotation (rotated: K % 256 == 0) undone as in filterPrepShader. K % 8 == 0.
// One workgroup per (256 columns, row n); each thread decodes 8 columns through the GEMM's loader.
export const denseShader = ({ b, vec, rotated }) => memo(`dense${b}${vec}${rotated}`, () => {
  const scaled = b === "i8" || b === "w4";
  const bindings = [];
  const bind = (decl) => bindings.push(`@group(0) @binding(${bindings.length + 1}) ${decl};`);
  bind(`var<storage, read> B: array<${weightArrayType(b, vec)}>`);
  if (scaled) bind("var<storage, read> RS: array<f32>");
  if (b === "w4") { bind("var<storage, read> SR: array<u32>"); bind("var<storage, read> CB: array<f32>"); }
  bind("var<storage, read_write> U: array<vec4<u32>>");
  const lanes = ["lo.x", "lo.y", "lo.z", "lo.w", "hi.x", "hi.y", "hi.z", "hi.w"];
  const pack = (j) => `pack2x16float(vec2<f32>(S[o + ${j}u], S[o + ${j + 1}u]) * norm)`;
  return /* wgsl */ `
struct Params { N: u32, K: u32, ldb: u32, bOff: u32, bBatch: u32, bDiv: u32, p0: u32, p1: u32 };
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${bindings.join("\n")}
${loaderB(b, vec)}
var<workgroup> S: array<f32, 256>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = wg.y;
  let k = wg.x * 256u + t * 8u;
  let o = t * 8u;
  let v = loadB8(0u, n, k);
  let s = ${scaled ? "RS[n]" : "1.0"};
  ${lanes.map((l, j) => `S[o + ${j}u] = v.${l} * s;`).join(" ")}
  workgroupBarrier();
  ${rotated ? `// undo ConvRot: x @ H, H = kron(H4,H4,H4,H4)/16
  for (var stride = 1u; stride < 256u; stride *= 4u) {
    for (var q = t; q < 64u; q += 32u) {
      let i0 = (q / stride) * stride * 4u + q % stride;
      let a = S[i0]; let b = S[i0 + stride]; let c = S[i0 + 2u * stride]; let e = S[i0 + 3u * stride];
      S[i0] = a + b + c - e;
      S[i0 + stride] = a + b - c + e;
      S[i0 + 2u * stride] = a - b + c + e;
      S[i0 + 3u * stride] = -a + b + c + e;
    }
    workgroupBarrier();
  }` : ""}
  let norm = ${rotated ? "0.0625" : "1.0"};
  if (k < P.K) { U[(n * P.K + k) / 8u] = vec4<u32>(${[0, 2, 4, 6].map(pack).join(", ")}); }
}`;
});

// Output transform: M [16][T][N] of tiles t0.. -> 2x2 tiles of NHWC Y [B, OH, OW, N] (+bias, act
// "none" | "leaky" (slope) | "relu"); bias: false for convs without one. One thread per (tile, 4 channels).
export const winogradOutShader = ({ act = "none", bias = true }) => memo(`wgout${act}${bias}`, () => {
  const m = (i, j) => `m${i}${j}`;
  const loads = [];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) loads.push(`let ${m(i, j)} = M[(${i * 4 + j}u * P.T + tl) * N4 + n4];`);
  // A^T = [[1,1,1,0],[0,1,-1,-1]]
  const t = [];
  for (let j = 0; j < 4; j++) {
    t.push(`let t0${j} = ${m(0, j)} + ${m(1, j)} + ${m(2, j)};`);
    t.push(`let t1${j} = ${m(1, j)} - ${m(2, j)} - ${m(3, j)};`);
  }
  const ys = [];
  for (let i = 0; i < 2; i++) {
    ys.push(`var y${i}0 = t${i}0 + t${i}1 + t${i}2 + bias;`);
    ys.push(`var y${i}1 = t${i}1 - t${i}2 - t${i}3 + bias;`);
  }
  const act4 = (v) => (act === "leaky" ? `${v} = select(${v} * P.slope, ${v}, ${v} >= vec4<f32>(0.0));`
    : act === "relu" ? `${v} = max(${v}, vec4<f32>(0.0));` : "");
  const stores = [];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      stores.push(`${act4(`y${i}${j}`)}\n  if (oy + ${i}u < P.OH && ox + ${j}u < P.OW) { Y[((b * P.OH + oy + ${i}u) * P.OW + ox + ${j}u) * N4 + n4] = y${i}${j}; }`);
    }
  }
  return /* wgsl */ `
struct Params { OH: u32, OW: u32, N: u32, th: u32, tw: u32, T: u32, t0: u32, nx: u32, slope: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> M: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;
${bias ? "@group(0) @binding(3) var<storage, read> BIAS: array<vec4<f32>>;" : ""}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N4 = P.N / 4u;
  let idx = gid.y * P.nx * 64u + gid.x;
  if (idx >= P.T * N4) { return; }
  let n4 = idx % N4;
  let tl = idx / N4;
  let tg = P.t0 + tl;
  let tx = tg % P.tw;
  let r = tg / P.tw;
  let b = r / P.th;
  let oy = 2u * (r % P.th);
  let ox = 2u * tx;
  let bias = ${bias ? "BIAS[n4]" : "vec4<f32>(0.0)"};
  ${loads.join("\n  ")}
  ${t.join("\n  ")}
  ${ys.join("\n  ")}
  ${stores.join("\n  ")}
}`;
});

// Direct k x k convolution to very few output channels (inswapper's last layer: 7x7, 128 -> 3,
// reflect padding), where a GEMM tile would mostly compute padding. NHWC X [B, H, W, C] ->
// Y [B, H, W, NO], + bias, then act "tanh01" ((tanh + 1) / 2), "sigmoid", "relu" or "none". pad
// "reflect" | "zeros".
// A workgroup computes a 16x16 tile of output pixels; the (16 + k - 1)^2 input patch is staged in
// workgroup memory 4 channels at a time. Weights are f32 [NO][k*k][C] (OHWI).
export const convFewShader = (k, NO, { pad = "reflect", act = "tanh01" } = {}) => memo(`convfew${k}x${NO}${pad}${act}`, () => {
  const P = 16 + k - 1;
  const r = (k - 1) / 2;
  const acc = [...Array(NO).keys()];
  const stage = pad === "reflect"
    ? `let sy = u32(refl(y0 + i32(i / ${P}u) - ${r}, i32(P.H)));
      let sx = u32(refl(x0 + i32(i % ${P}u) - ${r}, i32(P.W)));
      xs[i] = X[((b * P.H + sy) * P.W + sx) * C4 + c4];`
    : `let sy = y0 + i32(i / ${P}u) - ${r};
      let sx = x0 + i32(i % ${P}u) - ${r};
      if (sy >= 0 && sx >= 0 && sy < i32(P.H) && sx < i32(P.W)) { xs[i] = X[((b * P.H + u32(sy)) * P.W + u32(sx)) * C4 + c4]; } else { xs[i] = vec4<f32>(0.0); }`;
  const out = (o) => (act === "tanh01" ? `1.0 / (1.0 + exp(-2.0 * (a${o} + BIAS[${o}u])))` : act === "sigmoid" ? `1.0 / (1.0 + exp(-(a${o} + BIAS[${o}u])))`
    : act === "relu" ? `max(a${o} + BIAS[${o}u], 0.0)` : `a${o} + BIAS[${o}u]`);
  return /* wgsl */ `
struct Params { H: u32, W: u32, C: u32, tilesX: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> WT: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> BIAS: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> xs: array<vec4<f32>, ${P * P}>;
var<workgroup> ws: array<vec4<f32>, ${NO * k * k}>;
fn refl(i: i32, n: i32) -> i32 { return select(select(i, 2 * n - 2 - i, i >= n), -i, i < 0); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let b = wg.y;
  let x0 = i32(wg.x % P.tilesX) * 16;
  let y0 = i32(wg.x / P.tilesX) * 16;
  let lx = t % 16u;
  let ly = t / 16u;
  let C4 = P.C / 4u;
  ${acc.map((o) => `var a${o} = 0.0;`).join(" ")}
  for (var c4 = 0u; c4 < C4; c4++) {
    for (var i = t; i < ${P * P}u; i += 256u) {
      ${stage}
    }
    for (var i = t; i < ${NO * k * k}u; i += 256u) { ws[i] = WT[i * C4 + c4]; }
    workgroupBarrier();
    for (var ky = 0u; ky < ${k}u; ky++) {
      for (var kx = 0u; kx < ${k}u; kx++) {
        let v = xs[(ly + ky) * ${P}u + lx + kx];
        let tap = ky * ${k}u + kx;
        ${acc.map((o) => `a${o} += dot(v, ws[${o * k * k}u + tap]);`).join(" ")}
      }
    }
    workgroupBarrier();
  }
  let x = u32(x0) + lx;
  let y = u32(y0) + ly;
  if (x >= P.W || y >= P.H) { return; }
  let o = ((b * P.H + y) * P.W + x) * ${NO}u;
  ${acc.map((o) => `Y[o + ${o}u] = ${out(o)};`).join("\n  ")}
}`;
});

// ----------------------------------------------------------------------------- Enhancer layers
//
// StyleGAN's FIR filter is [1, 3, 3, 1] per axis: / 8 for the encoder's blur, / 4 (2D gain 4)
// after a zero-inserting 2x upsample.

// The encoder's blur before each stride-2 conv: X [B, H, W, C] -> Y [B, H+1, W+1, C], the 4x4
// filter over X zero-padded by 2. One thread per (pixel, 4 channels).
export const firDownShader = () => memo("firdown", () => /* wgsl */ `
struct Params { n4: u32, C4: u32, H: u32, W: u32, nx: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;
fn K(t: u32) -> f32 { return select(0.375, 0.125, t == 0u || t == 3u); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n4) { return; }
  let c = i % P.C4;
  let pix = i / P.C4;
  let x = pix % (P.W + 1u);
  let r = pix / (P.W + 1u);
  let y = r % (P.H + 1u);
  let b = r / (P.H + 1u);
  var acc = vec4<f32>(0.0);
  for (var ty = 0u; ty < 4u; ty++) {
    let sy = i32(y + ty) - 2;
    if (sy < 0 || sy >= i32(P.H)) { continue; }
    for (var tx = 0u; tx < 4u; tx++) {
      let sx = i32(x + tx) - 2;
      if (sx < 0 || sx >= i32(P.W)) { continue; }
      acc += X[((b * P.H + u32(sy)) * P.W + u32(sx)) * P.C4 + c] * (K(ty) * K(tx));
    }
  }
  Y[i] = acc;
}`);

// A modulated conv's output through the enhancer's activation: Y [B, H, W, 2C] =
// LeakyReLU_0.2([y*d | noise*F] + ACT), with y the conv output [B, H, W, C], d = sqrt 2 /
// sqrt(D + 1e-8) its demodulation (D [B, C] = s^2 @ demod; the scaled LeakyReLU's sqrt 2, which
// the stored ACT and noise already carry), F the encoder's feature map [B, H, W, C].
// up: y is the transposed conv's output blurred by the [1,3,3,1] filter (gain 4, zero pad 1),
// with the transposed conv held polyphase in X [B, H/2 + 1, W/2 + 1, 4C]: its pixel (2a + py,
// 2b + px) is X pixel (a, b), channels (py*2 + px)*C.. (models/enhancer.js). One thread per (pixel,
// 4 channels).
export const styledActShader = (up) => memo(`styledact${up}`, () => {
  const conv = up
    ? `
    let UH = P.H / 2u + 1u;
    let UW = P.W / 2u + 1u;
    v = vec4<f32>(0.0);
    for (var ty = 0u; ty < 4u; ty++) {
      let my = i32(y + ty) - 1;
      if (my < 0) { continue; }
      for (var tx = 0u; tx < 4u; tx++) {
        let mx = i32(x + tx) - 1;
        if (mx < 0) { continue; }
        let q = (u32(my) & 1u) * 2u + (u32(mx) & 1u);
        v += X[((b * UH + u32(my) / 2u) * UW + u32(mx) / 2u) * 4u * P.C4 + q * P.C4 + c] * (K(ty) * K(tx));
      }
    }`
    : `
    v = X[pix * P.C4 + c];`;
  return /* wgsl */ `
struct Params { n4: u32, C4: u32, H: u32, W: u32, nx: u32, noise: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> D: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> ACT: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> F: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> Y: array<vec4<f32>>;
fn K(t: u32) -> f32 { return select(0.75, 0.25, t == 0u || t == 3u); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n4) { return; }
  let c = i % (2u * P.C4);
  let pix = i / (2u * P.C4);         // (b * H + y) * W + x
  let x = pix % P.W;
  let y = (pix / P.W) % P.H;
  let b = pix / (P.W * P.H);
  var v: vec4<f32>;
  if (c < P.C4) {${conv}
    v = v * (inverseSqrt(D[b * P.C4 + c] + 1e-8) * 1.4142135) + ACT[c];
  } else {
    v = P.noise * F[pix * P.C4 + c - P.C4] + ACT[c];
  }
  Y[i] = select(v * 0.2, v, v >= vec4<f32>(0.0));
}`;
});

// the enhancer's modulated 1x1 conv to RGB: OUT [B, H, W, 3] = (X * s) @ WT^T + BIAS, plus with skip the
// 2x upsample of SKIP [B, H/2, W/2, 3] (zero insertion and the [1,3,3,1] filter, i.e. weights
// 1/4, 3/4 per axis with zeros outside). X [B, H, W, C], s the image's style at STY[styOff +
// b*styStride ..] (in vec4s), WT f32 [3][C]. One thread per pixel.
export const toRgbShader = (skip) => memo(`torgb${skip}`, () => /* wgsl */ `
struct Params { n: u32, C4: u32, H: u32, W: u32, nx: u32, styOff: u32, styStride: u32, p0: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> STY: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WT: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> BIAS: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
${skip ? `@group(0) @binding(6) var<storage, read> SKIP: array<f32>;
fn sk(b: u32, yy: i32, xx: i32, ch: u32) -> f32 {
  let h = i32(P.H / 2u);
  let w = i32(P.W / 2u);
  if (yy < 0 || xx < 0 || yy >= h || xx >= w) { return 0.0; }
  return SKIP[((b * u32(h) + u32(yy)) * u32(w) + u32(xx)) * 3u + ch];
}` : ""}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.y * P.nx * 256u + gid.x;
  if (p >= P.n) { return; }
  let b = p / (P.H * P.W);
  let sb = P.styOff + b * P.styStride;
  var a0 = 0.0;
  var a1 = 0.0;
  var a2 = 0.0;
  for (var c = 0u; c < P.C4; c++) {
    let v = X[p * P.C4 + c] * STY[sb + c];
    a0 += dot(v, WT[c]);
    a1 += dot(v, WT[P.C4 + c]);
    a2 += dot(v, WT[2u * P.C4 + c]);
  }
  var o = vec3<f32>(a0 + BIAS[0], a1 + BIAS[1], a2 + BIAS[2]);${skip ? `
  let x = p % P.W;
  let y = (p / P.W) % P.H;
  let y0 = (i32(y) - 1) >> 1u;       // floor((y - 1) / 2)
  let x0 = (i32(x) - 1) >> 1u;
  let wy = select(0.25, 0.75, (y & 1u) == 0u);  // weight of row y0 + 1
  let wx = select(0.25, 0.75, (x & 1u) == 0u);
  for (var ch = 0u; ch < 3u; ch++) {
    o[ch] += mix(mix(sk(b, y0, x0, ch), sk(b, y0, x0 + 1, ch), wx), mix(sk(b, y0 + 1, x0, ch), sk(b, y0 + 1, x0 + 1, ch), wx), wy);
  }` : ""}
  OUT[p * 3u] = o.x;
  OUT[p * 3u + 1u] = o.y;
  OUT[p * 3u + 2u] = o.z;
}`);

// Pixel norm of each row of X [rows, C]: Y = X / sqrt(mean(X^2) + 1e-8). One thread per row.
export const pixelNormShader = () => memo("pixnorm", () => /* wgsl */ `
struct Params { rows: u32, C: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= P.rows) { return; }
  var s = 0.0;
  for (var c = 0u; c < P.C; c++) { let v = X[r * P.C + c]; s += v * v; }
  let k = inverseSqrt(s / f32(P.C) + 1e-8);
  for (var c = 0u; c < P.C; c++) { Y[r * P.C + c] = X[r * P.C + c] * k; }
}`);

// Y = X * X, elementwise
export const squareShader = () => memo("square", () => /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = X[i] * X[i]; }
}`);

// The enhancer's detail band (models/enhancer.js): the part of C - G (enhancer input and output,
// [B, N, N, 3]) below LO px, shrunk to LO x LO: an n x n average of bilinear samples (edges
// replicated) around crop position s * (x, y), s = N / LO. One thread
// per output pixel.
export const bandDownShader = () => memo("banddown", () => /* wgsl */ `
struct Params { N: u32, LO: u32, n: u32, p0: u32, s: f32, p1: u32, p2: u32, p3: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> C: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> D: array<f32>;
fn px(b: u32, x: i32, y: i32) -> vec3<f32> {
  let m = i32(P.N) - 1;
  let i = ((b * P.N + u32(clamp(y, 0, m))) * P.N + u32(clamp(x, 0, m))) * 3u;
  return vec3<f32>(C[i] - G[i], C[i + 1u] - G[i + 1u], C[i + 2u] - G[i + 2u]);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  let b = gid.z;
  if (ox >= P.LO || oy >= P.LO) { return; }
  let n = f32(P.n);
  var acc = vec3<f32>(0.0);
  for (var j = 0u; j < P.n; j++) {
    for (var i = 0u; i < P.n; i++) {
      let sx = P.s * (f32(ox) + (f32(i) + 0.5) / n - 0.5);
      let sy = P.s * (f32(oy) + (f32(j) + 0.5) / n - 0.5);
      let fx = floor(sx);
      let fy = floor(sy);
      let x0 = i32(fx);
      let y0 = i32(fy);
      acc += mix(mix(px(b, x0, y0), px(b, x0 + 1, y0), sx - fx), mix(px(b, x0, y0 + 1), px(b, x0 + 1, y0 + 1), sx - fx), sy - fy);
    }
  }
  let o = ((b * P.LO + oy) * P.LO + ox) * 3u;
  let v = acc / (n * n);
  D[o] = v.x;
  D[o + 1u] = v.y;
  D[o + 2u] = v.z;
}`);

// The enhancer's output for the paste, [-1, 1] -> [0, 1]: OUT = (G + 1) / 2, or with band its
// change to the input crop C, (G + up(D) - C) / 2, up(D) the LO x LO detail band D bilinearly
// enlarged to N (crop position (x, y) * LO / N, edges replicated). One thread per pixel.
export const enhanceOutShader = (band) => memo(`enhout${band}`, () => /* wgsl */ `
struct Params { n: u32, N: u32, LO: u32, nx: u32, r: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
${band ? `@group(0) @binding(3) var<storage, read> D: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
fn dp(b: u32, x: i32, y: i32) -> vec3<f32> {
  let m = i32(P.LO) - 1;
  let i = ((b * P.LO + u32(clamp(y, 0, m))) * P.LO + u32(clamp(x, 0, m))) * 3u;
  return vec3<f32>(D[i], D[i + 1u], D[i + 2u]);
}` : ""}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.y * P.nx * 256u + gid.x;
  if (p >= P.n) { return; }
  var v = vec3<f32>(G[p * 3u], G[p * 3u + 1u], G[p * 3u + 2u]);${band ? `
  let x = p % P.N;
  let y = (p / P.N) % P.N;
  let b = p / (P.N * P.N);
  let sx = f32(x) * P.r;
  let sy = f32(y) * P.r;
  let fx = floor(sx);
  let fy = floor(sy);
  let x0 = i32(fx);
  let y0 = i32(fy);
  v += mix(mix(dp(b, x0, y0), dp(b, x0 + 1, y0), sx - fx), mix(dp(b, x0, y0 + 1), dp(b, x0 + 1, y0 + 1), sx - fx), sy - fy);
  v = (v - vec3<f32>(C[p * 3u], C[p * 3u + 1u], C[p * 3u + 2u])) * 0.5;` : `
  v = v * 0.5 + 0.5;`}
  OUT[p * 3u] = v.x;
  OUT[p * 3u + 1u] = v.y;
  OUT[p * 3u + 2u] = v.z;
}`);

// A face's radial power spectrum (ops.faceSpectrum), in three steps over B crops X [B, N, N, 3]
// (values 0..1; N = 256, one thread per pixel of a row or column):
// rows: the DFT along x of each row's luma less 1/2 (which only changes bins 0 and 1, and keeps
// them from swamping the rest in f32), Hann-windowed in x and y -> F [B, N, N, 2] (row y,
// frequency u; re, im). One workgroup per row: (y, b).
export const spectrumRowsShader = (N) => memo(`specrows${N}`, () => /* wgsl */ `
struct Params { b0: u32, p1: u32, p2: u32, p3: u32 }; // b0: the first crop (0)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> F: array<f32>;
const N = ${N}u;
const TAU = 6.283185307179586;
var<workgroup> row: array<f32, N>;
fn hann(i: u32) -> f32 { return 0.5 - 0.5 * cos(TAU * f32(i) / f32(N - 1u)); }
@compute @workgroup_size(${N})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let y = wg.x;
  let b = P.b0 + wg.y;
  let o = ((b * N + y) * N + t) * 3u;
  row[t] = (0.299 * X[o] + 0.587 * X[o + 1u] + 0.114 * X[o + 2u] - 0.5) * hann(t) * hann(y);
  workgroupBarrier();
  var re = 0.0;
  var im = 0.0;
  for (var x = 0u; x < N; x++) {
    let a = TAU * f32((t * x) % N) / f32(N);
    re += row[x] * cos(a);
    im -= row[x] * sin(a);
  }
  let q = ((b * N + y) * N + t) * 2u;
  F[q] = re;
  F[q + 1u] = im;
}`);

// cols: the DFT along y of each column of F, and its power -> PW [B, N, N] (frequency v, u). One
// workgroup per column: (u, b).
export const spectrumColsShader = (N) => memo(`speccols${N}`, () => /* wgsl */ `
struct Params { b0: u32, p1: u32, p2: u32, p3: u32 }; // b0: the first crop (0)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read_write> PW: array<f32>;
const N = ${N}u;
const TAU = 6.283185307179586;
var<workgroup> col: array<vec2<f32>, N>;
@compute @workgroup_size(${N})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let u = wg.x;
  let b = P.b0 + wg.y;
  let q = ((b * N + t) * N + u) * 2u;
  col[t] = vec2<f32>(F[q], F[q + 1u]);
  workgroupBarrier();
  var re = 0.0;
  var im = 0.0;
  for (var y = 0u; y < N; y++) {
    let a = TAU * f32((t * y) % N) / f32(N);
    let c = cos(a);
    let s = sin(a);
    re += col[y].x * c + col[y].y * s;
    im += col[y].y * c - col[y].x * s;
  }
  PW[(b * N + t) * N + u] = re * re + im * im;
}`);

// radial: PW summed per radial frequency bin round(sqrt(u'^2 + v'^2)), u' and v' the signed
// frequencies (|u'| = min(u, N - u)), bins 0..N/2 -> R [B, N/2 + 1]. One workgroup per (bin, b),
// its threads over every N-th pixel, then a tree sum.
export const spectrumRadialShader = (N) => memo(`specrad${N}`, () => /* wgsl */ `
struct Params { b0: u32, p1: u32, p2: u32, p3: u32 }; // b0: the first crop (0)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> PW: array<f32>;
@group(0) @binding(2) var<storage, read_write> R: array<f32>;
const N = ${N}u;
var<workgroup> acc: array<f32, N>;
@compute @workgroup_size(${N})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let bin = wg.x;
  let b = P.b0 + wg.y;
  var s = 0.0;
  for (var v = 0u; v < N; v++) {
    let av = min(v, N - v);
    let au = min(t, N - t);
    let r = min(u32(round(sqrt(f32(au * au + av * av)))), N / 2u);
    if (r == bin) { s += PW[(b * N + v) * N + t]; }
  }
  acc[t] = s;
  workgroupBarrier();
  for (var w = N / 2u; w > 0u; w /= 2u) {
    if (t < w) { acc[t] += acc[t + w]; }
    workgroupBarrier();
  }
  if (t == 0u) { R[b * (N / 2u + 1u) + bin] = acc[0]; }
}`);

// ----------------------------------------------------------------------------- XSeg layers
//
// FRN + TLU (Filter Response Normalization with a thresholded linear unit) after every conv:
// y = max(gamma * x / sqrt(nu2 + eps) + beta, tau), nu2 the mean of x^2 over the image's pixels,
// per (image, channel); eps one scalar per layer.

// FRN statistics folded into the affine frnOutShader applies: for X [B, HW, G*C] (G > 1: the
// phase groups of a polyphase transposed conv, which are the same channel), with FRN = [gamma (C),
// beta (C), tau (C)],
//   S = gamma / sqrt(nu2 + eps), T = beta, U = tau       (PRO = [S (B*C), T (B*C), U (B*C)])
// One workgroup per (64 channels, image): 16 vec4 channel columns x 16 pixel lanes.
export const frnShader = () => memo("frn", () => /* wgsl */ `
struct Params { HW: u32, C: u32, B: u32, G: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> FRN: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> PRO: array<vec4<f32>>;
var<workgroup> red: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let b = wg.y;
  let lane = t / 16u;
  let c4 = wg.x * 16u + t % 16u;
  let C4 = P.C / 4u;
  let GC4 = P.G * C4;
  var s = vec4<f32>(0.0);
  if (c4 < C4) {
    for (var p = lane; p < P.HW; p += 16u) {
      for (var g = 0u; g < P.G; g++) { let v = X[(b * P.HW + p) * GC4 + g * C4 + c4]; s += v * v; }
    }
  }
  red[t] = s;
  workgroupBarrier();
  for (var k = 128u; k >= 16u; k >>= 1u) {
    if (t < k) { red[t] = red[t] + red[t + k]; }
    workgroupBarrier();
  }
  if (lane == 0u && c4 < C4) {
    let o = b * C4 + c4;
    let BC4 = P.B * C4;
    PRO[o] = FRN[c4] * inverseSqrt(red[t] / f32(P.HW * P.G) + P.eps);
    PRO[BC4 + o] = FRN[C4 + c4];
    PRO[2u * BC4 + o] = FRN[2u * C4 + c4];
  }
}`);

// The same statistics for large images, in two steps: frnPartsShader sums x^2 over each (image,
// chunk of pixels, 64 of the G*C channels), frnCombineShader adds up the chunks and the groups and
// writes PRO as frnShader. PART: [B][chunks][G*C].
export const frnPartsShader = () => memo("frnparts", () => /* wgsl */ `
struct Params { HW: u32, C: u32, chunk: u32, chunks: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> PART: array<vec4<f32>>;
var<workgroup> red: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let b = wg.z;
  let k = wg.y;
  let lane = t / 16u;
  let c4 = wg.x * 16u + t % 16u;
  let C4 = P.C / 4u;
  let p1 = min((k + 1u) * P.chunk, P.HW);
  var s = vec4<f32>(0.0);
  if (c4 < C4) {
    for (var p = k * P.chunk + lane; p < p1; p += 16u) { let v = X[(b * P.HW + p) * C4 + c4]; s += v * v; }
  }
  red[t] = s;
  workgroupBarrier();
  for (var j = 128u; j >= 16u; j >>= 1u) {
    if (t < j) { red[t] = red[t] + red[t + j]; }
    workgroupBarrier();
  }
  if (lane == 0u && c4 < C4) { PART[(b * P.chunks + k) * C4 + c4] = red[t]; }
}`);

export const frnCombineShader = () => memo("frncombine", () => /* wgsl */ `
struct Params { HW: u32, C: u32, B: u32, G: u32, chunks: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> PART: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> FRN: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> PRO: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let C4 = P.C / 4u;
  if (gid.x >= P.B * C4) { return; }
  let b = gid.x / C4;
  let c4 = gid.x % C4;
  let GC4 = P.G * C4;
  var s = vec4<f32>(0.0);
  for (var k = 0u; k < P.chunks; k++) {
    for (var g = 0u; g < P.G; g++) { s += PART[(b * P.chunks + k) * GC4 + g * C4 + c4]; }
  }
  let BC4 = P.B * C4;
  PRO[gid.x] = FRN[c4] * inverseSqrt(s / f32(P.HW * P.G) + P.eps);
  PRO[BC4 + gid.x] = FRN[C4 + c4];
  PRO[2u * BC4 + gid.x] = FRN[2u * C4 + c4];
}`);

// A conv's output through its FRN + TLU, written for its consumer: Y [B, OH, OW, C + CS] =
// [max(X*S + T, U) | SKIP], PRO = [S, T, U] from frnShader.
//   shuffle: X is a transposed conv held polyphase, [B, OH/2, OW/2, 4C]: Y pixel (2a + py,
//            2b + px) is X pixel (a, b), channels (py*2 + px)*C.. (weights.js polyphase);
//            otherwise X is [B, OH, OW, C]
//   skip:    SKIP [B, OH, OW, CS] is appended (the decoder's concat with the encoder's map)
//   rotate:  Y is ConvRot-rotated (the input of a quantized conv; C + CS a multiple of 256): one
//            workgroup of 64 threads per 256 channels of a pixel. Otherwise one thread per (pixel,
//            4 channels).
export const frnOutShader = ({ shuffle = false, skip = false, rotate = false }) => memo(`frnout${shuffle}${skip}${rotate}`, () => {
  const src = shuffle
    ? `let x = pix % P.OW;
  let y = (pix / P.OW) % P.OH;
  let src = ((b * (P.OH / 2u) + y / 2u) * (P.OW / 2u) + x / 2u) * 4u * P.C4 + ((y & 1u) * 2u + (x & 1u)) * P.C4 + c;`
    : "let src = pix * P.C4 + c;";
  return /* wgsl */ `
struct Params { n4: u32, C4: u32, CS4: u32, OH: u32, OW: u32, B: u32, nx: u32, p0: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> PRO: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
${skip ? "@group(0) @binding(4) var<storage, read> SKIP: array<vec4<f32>>;" : ""}
fn value(i: u32) -> vec4<f32> {
  let c = i % (P.C4 + P.CS4);
  let pix = i / (P.C4 + P.CS4);
  ${skip ? "if (c >= P.C4) { return SKIP[pix * P.CS4 + c - P.C4]; }" : ""}
  let b = pix / (P.OH * P.OW);
  ${src}
  let s = b * P.C4 + c;
  let BC4 = P.B * P.C4;
  return max(X[src] * PRO[s] + PRO[BC4 + s], PRO[2u * BC4 + s]);
}
${rotate ? `var<workgroup> S: array<f32, 256>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.y * P.nx + wg.x;
  if (g >= P.n4 / 64u) { return; }
  let v = value(g * 64u + t);
  S[4u * t] = v.x; S[4u * t + 1u] = v.y; S[4u * t + 2u] = v.z; S[4u * t + 3u] = v.w;
  workgroupBarrier();
  var stride = 1u;
  for (var d = 0u; d < 4u; d++) {
    let i0 = (t / stride) * stride * 4u + t % stride;
    let a = S[i0]; let b = S[i0 + stride]; let c = S[i0 + 2u * stride]; let e = S[i0 + 3u * stride];
    workgroupBarrier();
    S[i0] = a + b + c - e;
    S[i0 + stride] = a + b - c + e;
    S[i0 + 2u * stride] = a - b + c + e;
    S[i0 + 3u * stride] = -a + b + c + e;
    workgroupBarrier();
    stride *= 4u;
  }
  Y[g * 64u + t] = vec4<f32>(S[4u * t], S[4u * t + 1u], S[4u * t + 2u], S[4u * t + 3u]) * 0.0625;
}` : `@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n4) { Y[i] = value(i); }
}`}`;
});

// The encoder's blurred stride-2 downsampling, X [B, H, W, C] -> Y [B, H/2, W/2, C]: the separable
// filter TAPS (k = 4: [1,3,3,1] / 8, k = 3: [1,2,1] / 4 per axis) over X zero-padded by 1 before,
// output pixel o reading input pixels 2o - 1 ... One thread per (pixel, 4 channels).
export const blurDownShader = (taps) => memo(`blurdown${taps}`, () => {
  const sum = taps.reduce((a, b) => a + b, 0);
  const wt = (t) => `${taps[t] / sum}`;
  // unrolled: the taps are constants
  const rows = taps.map((_, ty) => `
  { let sy = 2 * i32(y) + ${ty - 1};
    if (sy >= 0 && sy < i32(P.H)) {
      let r = (b * P.H + u32(sy)) * P.W;
      var row = vec4<f32>(0.0);${taps.map((__, tx) => `
      { let sx = 2 * i32(x) + ${tx - 1}; if (sx >= 0 && sx < i32(P.W)) { row += X[(r + u32(sx)) * P.C4 + c] * ${wt(tx)}; } }`).join("")}
      acc += row * ${wt(ty)};
    } }`).join("");
  return /* wgsl */ `
struct Params { n4: u32, C4: u32, H: u32, W: u32, nx: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n4) { return; }
  let c = i % P.C4;
  let pix = i / P.C4;
  let x = pix % (P.W / 2u);
  let y = (pix / (P.W / 2u)) % (P.H / 2u);
  let b = pix / ((P.W / 2u) * (P.H / 2u));
  var acc = vec4<f32>(0.0);${rows}
  Y[i] = acc;
}`;
});

// NHWC X [B, HW, C] -> NCHW Y [B, C, HW] (a flatten in the original model's order). One thread
// per element.
export const nchwShader = () => memo("nchw", () => /* wgsl */ `
struct Params { n: u32, HW: u32, C: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let p = i % P.HW;
  let c = (i / P.HW) % P.C;
  let b = i / (P.HW * P.C);
  Y[i] = X[(b * P.HW + p) * P.C + c];
}`);

// Grows (the most in a disk of radius r around each pixel) or, with erode, shrinks (the least)
// what's bright in single-channel images X [B, N, N]. What's outside an image doesn't count.
export const maskMorphShader = (erode) => memo(`maskmorph${erode}`, () => /* wgsl */ `
struct Params { n: u32, N: u32, r: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let N = i32(P.N);
  let r = i32(P.r);
  let x = i32(i % P.N);
  let y = i32((i / P.N) % P.N);
  let base = i32(i) - y * N - x;
  var v = X[i];
  for (var yy = max(y - r, 0); yy <= min(y + r, N - 1); yy++) {
    let w = i32(sqrt(f32(r * r - (yy - y) * (yy - y))));
    for (var xx = max(x - w, 0); xx <= min(x + w, N - 1); xx++) {
      v = ${erode ? "min" : "max"}(v, X[base + yy * N + xx]);
    }
  }
  Y[i] = v;
}`);

// One axis of a Gaussian blur on single-channel images X [B, N, N] (cv2.GaussianBlur's
// BORDER_REFLECT_101), with the normalized kernel TAPS [2r + 1]. finish: then the
// occlusion mask from the blurred XSeg output, (clamp(v, 0.5, 1) - 0.5) * 2. One thread per pixel.
export const maskBlurShader = (vertical, finish) => memo(`maskblur${vertical}${finish}`, () => /* wgsl */ `
struct Params { n: u32, N: u32, r: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> TAPS: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i >= P.n) { return; }
  let N = i32(P.N);
  let pos = i32(${vertical ? "(i / P.N) % P.N" : "i % P.N"});
  let base = i - u32(pos) * ${vertical ? "P.N" : "1u"};
  var v = 0.0;
  for (var t = 0u; t <= 2u * P.r; t++) {
    var j = abs(pos + i32(t) - i32(P.r));
    if (j >= N) { j = 2 * (N - 1) - j; }
    v += X[base + u32(j) * ${vertical ? "P.N" : "1u"}] * TAPS[t];
  }
  Y[i] = ${finish ? "(clamp(v, 0.5, 1.0) - 0.5) * 2.0" : "v"};
}`);

// f32 -> packed f16 pairs (n even): Y[i] = pack2x16float(X[2i], X[2i + 1])
export const packHalfShader = () => memo("packhalf", () => /* wgsl */ `
struct Params { n2: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n2) { Y[i] = pack2x16float(X[i]); }
}`);

export const zeroShader = () => memo("zero", () => /* wgsl */ `
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = 0.0; }
}`);
