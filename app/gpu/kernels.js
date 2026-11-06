var w=Object.defineProperty;var i=(r,a)=>w(r,"name",{value:a,configurable:!0});const y=new Map,t=i((r,a)=>(y.has(r)||y.set(r,a()),y.get(r)),"memo");import{epilogue as $,loaderB as b,weightArrayType as m}from"./gemm.js";import{matmulShader as lr}from"./gemm.js";const S=i(({scaled:r=!1,bias:a=!1,add:o=!1,act:l="none",resid:n=!1})=>t(`splitred${r}${a}${o}${l}${n}`,()=>{const s=[],g=i(d=>s.push(`@group(0) @binding(${s.length+1}) ${d};`),"bind");return g("var<storage, read> PART: array<f32>"),g("var<storage, read_write> C: array<f32>"),r&&g("var<storage, read> RS: array<f32>"),a&&g("var<storage, read> BIAS: array<f32>"),o&&g("var<storage, read> ADD: array<f32>"),l==="prelu"&&g("var<storage, read> SL: array<f32>"),`
struct Params { M: u32, N: u32, S: u32, alpha: f32, nx: u32, slope: f32, ldc: u32, cOff: u32 };
@group(0) @binding(0) var<uniform> P: Params;
${s.join(`
`)}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  let MN = P.M * P.N;
  if (i >= MN) { return; }
  let m = i / P.N;
  let n = i % P.N;
  var r = 0.0;
  for (var s = 0u; s < P.S; s++) { r += PART[s * MN + i]; }
  r = r * P.alpha;${$({scaled:r,bias:a,add:o,act:l})}
  let ci = P.cOff + m * P.ldc + n;
  ${n?"C[ci] = C[ci] + r;":"C[ci] = r;"}
}`}),"splitReduceShader"),C=`
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
}`,h=i(()=>t("had",()=>`
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
}`),"hadamardShader"),W=i(()=>t("adain",()=>`
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
}`),"adainShader"),H=i(()=>t("normparts",()=>`
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
}`),"normPartsShader"),O=i(()=>t("normcombine",()=>`
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
}`),"normCombineShader"),N=i((r,a)=>t(`aff${r}${a}`,()=>`
struct Params { rows: u32, C: u32, HW: u32, proStride: u32, proOff: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> PRO: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
${a?`var<workgroup> S: array<f32, 2048>;
`+C:""}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let row = wg.y * P.nx + wg.x;
  if (row >= P.rows) { return; }
  let b = row / P.HW;
  let base = row * P.C;
  let pb = b * P.proStride;
  for (var c = t; c < P.C; c += 256u) {
    var v = ${r==="scale"?"X[base + c] * PRO[P.proOff + pb + c]":"X[base + c] * PRO[pb + c] + PRO[P.proOff + pb + c]"};
    ${r==="relu"?"v = max(v, 0.0);":""}
    ${a?"S[c] = v;":r==="resid"?"Y[base + c] = Y[base + c] + v;":"Y[base + c] = v;"}
  }
  ${a?`workgroupBarrier();
  hadamardRows(t, P.C);
  for (var c = t; c < P.C; c += 256u) { Y[base + c] = S[c] * 0.0625; }`:""}
}`),"affineShader"),R=i(r=>t("pool"+r,()=>`
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
  Y[i] = ${r==="max"?"max(max(a, b), max(d, e))":"(a + b + d + e) * 0.25"};
}`),"pool2Shader"),B=i(()=>t("upadd",()=>`
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
}`),"upAddShader"),X=i(()=>t("resample",()=>`
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
}`),"resampleShader"),z=i((r=!1)=>t(`paste${r}`,()=>`
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
  return ${r?"vec3<f32>(SW[o], SW[o + 1u], SW[o + 2u])":"clamp(vec3<f32>(SW[o], SW[o + 1u], SW[o + 2u]), vec3<f32>(0.0), vec3<f32>(1.0))"};
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
  FRAME[i] = pack4x8unorm(vec4<f32>(${r?"f + sw * a":"mix(f, sw, a)"}, 1.0));
}`),"pasteShader"),j=i(()=>t("unocclude",()=>`
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
}`),"unoccludeShader"),T=i(({pad:r="zeros",up:a="",pro:o=""})=>t(`wgin${r}${a}${o}`,()=>{const l=a?"2 * i32(P.IH)":"i32(P.IH)",n=a?"2 * i32(P.IW)":"i32(P.IW)",s=r==="reflect"?`y = refl(y, ${l}); x = refl(x, ${n});`:`if (y < 0 || x < 0 || y >= ${l} || x >= ${n}) { return vec4<f32>(0.0); }`;let g;a==="bilinear"?g=`
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
              mix(X[(r1 + x0) * C4 + c4], X[(r1 + x1) * C4 + c4], sx - f32(x0)), sy - f32(y0));`:g=`
  var v = X[((b * P.IH + ${a==="nearest"?"u32(y) / 2u":"u32(y)"}) * P.IW + ${a==="nearest"?"u32(x) / 2u":"u32(x)"}) * (P.C / 4u) + c4];`;const d=o==="scale"?`
  v = v * PRO[(P.proOff + b * P.proStride) / 4u + c4];`:o?`
  let si = (b * P.proStride) / 4u + c4;
  v = v * PRO[si] + PRO[P.proOff / 4u + si];${o==="affine_relu"?`
  v = max(v, vec4<f32>(0.0));`:""}`:"",e=i((c,p)=>`d${c}${p}`,"d"),u=[];for(let c=0;c<4;c++)for(let p=0;p<4;p++)u.push(`let ${e(c,p)} = px(b, y0 + ${c}, x0 + ${p}, c4);`);const P=i(c=>[`${c(0)} - ${c(2)}`,`${c(1)} + ${c(2)}`,`${c(2)} - ${c(1)}`,`${c(1)} - ${c(3)}`],"rows"),v=[];for(let c=0;c<4;c++)P(p=>e(p,c)).forEach((p,x)=>v.push(`let t${x}${c} = ${p};`));const f=[];for(let c=0;c<4;c++)P(p=>`t${c}${p}`).forEach((p,x)=>f.push(`V[(${c*4+x}u * P.T + tl) * C4 + c4] = ${p};`));return`
struct Params { IH: u32, IW: u32, C: u32, th: u32, tw: u32, T: u32, t0: u32, nx: u32, proOff: u32, proStride: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> V: array<vec4<f32>>;
${o?"@group(0) @binding(3) var<storage, read> PRO: array<vec4<f32>>;":""}
fn refl(i: i32, n: i32) -> i32 { return select(select(i, 2 * n - 2 - i, i >= n), -i, i < 0); }
fn px(b: u32, yy: i32, xx: i32, c4: u32) -> vec4<f32> {
  var y = yy;
  var x = xx;
  ${s}${g}${d}
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
  ${u.join(`
  `)}
  ${v.join(`
  `)}
  ${f.join(`
  `)}
}`}),"winogradInShader"),A=i(({b:r,vec:a,rotated:o,gs:l,out:n="winograd"})=>t(`fprep${r}${a}${o}${l}${n}`,()=>{const s=l/8,g=i(f=>`vec4<u32>(pack2x16float(vec2<f32>(${f(0)}, ${f(1)})), pack2x16float(vec2<f32>(${f(2)}, ${f(3)})),
      pack2x16float(vec2<f32>(${f(4)}, ${f(5)})), pack2x16float(vec2<f32>(${f(6)}, ${f(7)})))`,"pack8"),d=n==="ohwi"?`
  for (var tap = 0u; tap < 9u; tap++) {
    let o = tap * ${l}u + t * 8u;
    U[((n * 9u + tap) * P.C + c0) / 8u] = ${g(f=>`S[o + ${f}u] * norm`)};
  }`:`
  var u: array<array<f32, 8>, 16>;
  for (var j = 0u; j < 8u; j++) {
    let c = t * 8u + j;
    var g: array<f32, 9>;
    for (var tap = 0u; tap < 9u; tap++) { g[tap] = S[tap * ${l}u + c] * norm; }
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
    U[((p * P.N + n) * P.C + c0) / 8u] = ${g(f=>`u[p][${f}]`)};
  }`,e=r==="i8"||r==="w4",u=[],P=i(f=>u.push(`@group(0) @binding(${u.length+1}) ${f};`),"bind");P(`var<storage, read> B: array<${m(r,a)}>`),e&&P("var<storage, read> RS: array<f32>"),r==="w4"&&(P("var<storage, read> SR: array<u32>"),P("var<storage, read> CB: array<f32>")),P("var<storage, read_write> U: array<vec4<u32>>");const v=["lo.x","lo.y","lo.z","lo.w","hi.x","hi.y","hi.z","hi.w"];return`
struct Params { N: u32, K: u32, ldb: u32, bOff: u32, bBatch: u32, bDiv: u32, C: u32, p0: u32 };
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${u.join(`
`)}
${b(r,a)}
var<workgroup> S: array<f32, ${9*l}>;
@compute @workgroup_size(${s})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = wg.y;
  let c0 = wg.x * ${l}u + t * 8u;
  let s = ${e?"RS[n]":"1.0"};
  for (var tap = 0u; tap < 9u; tap++) {
    let v = loadB8(0u, n, tap * P.C + c0);
    let o = tap * ${l}u + t * 8u;
    ${v.map((f,c)=>`S[o + ${c}u] = v.${f} * s;`).join(" ")}
  }
  workgroupBarrier();
  ${o?`// undo ConvRot: x @ H per tap, H = kron(H4,H4,H4,H4)/16
  for (var stride = 1u; stride < 256u; stride *= 4u) {
    for (var q = t; q < 9u * 64u; q += ${s}u) {
      let i0 = (q / 64u) * 256u + ((q % 64u) / stride) * stride * 4u + (q % 64u) % stride;
      let a = S[i0]; let b = S[i0 + stride]; let c = S[i0 + 2u * stride]; let e = S[i0 + 3u * stride];
      S[i0] = a + b + c - e;
      S[i0 + stride] = a + b - c + e;
      S[i0 + 2u * stride] = a - b + c + e;
      S[i0 + 3u * stride] = -a + b + c + e;
    }
    workgroupBarrier();
  }`:""}
  let norm = ${o?"0.0625":"1.0"};${d}
}`}),"filterPrepShader"),Y=i(({b:r,vec:a,rotated:o})=>t(`dense${r}${a}${o}`,()=>{const l=r==="i8"||r==="w4",n=[],s=i(e=>n.push(`@group(0) @binding(${n.length+1}) ${e};`),"bind");s(`var<storage, read> B: array<${m(r,a)}>`),l&&s("var<storage, read> RS: array<f32>"),r==="w4"&&(s("var<storage, read> SR: array<u32>"),s("var<storage, read> CB: array<f32>")),s("var<storage, read_write> U: array<vec4<u32>>");const g=["lo.x","lo.y","lo.z","lo.w","hi.x","hi.y","hi.z","hi.w"],d=i(e=>`pack2x16float(vec2<f32>(S[o + ${e}u], S[o + ${e+1}u]) * norm)`,"pack");return`
struct Params { N: u32, K: u32, ldb: u32, bOff: u32, bBatch: u32, bDiv: u32, p0: u32, p1: u32 };
struct V8 { lo: vec4<f32>, hi: vec4<f32> };
@group(0) @binding(0) var<uniform> P: Params;
${n.join(`
`)}
${b(r,a)}
var<workgroup> S: array<f32, 256>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let n = wg.y;
  let k = wg.x * 256u + t * 8u;
  let o = t * 8u;
  let v = loadB8(0u, n, k);
  let s = ${l?"RS[n]":"1.0"};
  ${g.map((e,u)=>`S[o + ${u}u] = v.${e} * s;`).join(" ")}
  workgroupBarrier();
  ${o?`// undo ConvRot: x @ H, H = kron(H4,H4,H4,H4)/16
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
  }`:""}
  let norm = ${o?"0.0625":"1.0"};
  if (k < P.K) { U[(n * P.K + k) / 8u] = vec4<u32>(${[0,2,4,6].map(d).join(", ")}); }
}`}),"denseShader"),q=i(({act:r="none",bias:a=!0})=>t(`wgout${r}${a}`,()=>{const o=i((e,u)=>`m${e}${u}`,"m"),l=[];for(let e=0;e<4;e++)for(let u=0;u<4;u++)l.push(`let ${o(e,u)} = M[(${e*4+u}u * P.T + tl) * N4 + n4];`);const n=[];for(let e=0;e<4;e++)n.push(`let t0${e} = ${o(0,e)} + ${o(1,e)} + ${o(2,e)};`),n.push(`let t1${e} = ${o(1,e)} - ${o(2,e)} - ${o(3,e)};`);const s=[];for(let e=0;e<2;e++)s.push(`var y${e}0 = t${e}0 + t${e}1 + t${e}2 + bias;`),s.push(`var y${e}1 = t${e}1 - t${e}2 - t${e}3 + bias;`);const g=i(e=>r==="leaky"?`${e} = select(${e} * P.slope, ${e}, ${e} >= vec4<f32>(0.0));`:r==="relu"?`${e} = max(${e}, vec4<f32>(0.0));`:"","act4"),d=[];for(let e=0;e<2;e++)for(let u=0;u<2;u++)d.push(`${g(`y${e}${u}`)}
  if (oy + ${e}u < P.OH && ox + ${u}u < P.OW) { Y[((b * P.OH + oy + ${e}u) * P.OW + ox + ${u}u) * N4 + n4] = y${e}${u}; }`);return`
struct Params { OH: u32, OW: u32, N: u32, th: u32, tw: u32, T: u32, t0: u32, nx: u32, slope: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> M: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<vec4<f32>>;
${a?"@group(0) @binding(3) var<storage, read> BIAS: array<vec4<f32>>;":""}
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
  let bias = ${a?"BIAS[n4]":"vec4<f32>(0.0)"};
  ${l.join(`
  `)}
  ${n.join(`
  `)}
  ${s.join(`
  `)}
  ${d.join(`
  `)}
}`}),"winogradOutShader"),I=i((r,a,{pad:o="reflect",act:l="tanh01"}={})=>t(`convfew${r}x${a}${o}${l}`,()=>{const n=16+r-1,s=(r-1)/2,g=[...Array(a).keys()],d=o==="reflect"?`let sy = u32(refl(y0 + i32(i / ${n}u) - ${s}, i32(P.H)));
      let sx = u32(refl(x0 + i32(i % ${n}u) - ${s}, i32(P.W)));
      xs[i] = X[((b * P.H + sy) * P.W + sx) * C4 + c4];`:`let sy = y0 + i32(i / ${n}u) - ${s};
      let sx = x0 + i32(i % ${n}u) - ${s};
      if (sy >= 0 && sx >= 0 && sy < i32(P.H) && sx < i32(P.W)) { xs[i] = X[((b * P.H + u32(sy)) * P.W + u32(sx)) * C4 + c4]; } else { xs[i] = vec4<f32>(0.0); }`,e=i(u=>l==="tanh01"?`1.0 / (1.0 + exp(-2.0 * (a${u} + BIAS[${u}u])))`:l==="sigmoid"?`1.0 / (1.0 + exp(-(a${u} + BIAS[${u}u])))`:l==="relu"?`max(a${u} + BIAS[${u}u], 0.0)`:`a${u} + BIAS[${u}u]`,"out");return`
struct Params { H: u32, W: u32, C: u32, tilesX: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> WT: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> BIAS: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> xs: array<vec4<f32>, ${n*n}>;
var<workgroup> ws: array<vec4<f32>, ${a*r*r}>;
fn refl(i: i32, n: i32) -> i32 { return select(select(i, 2 * n - 2 - i, i >= n), -i, i < 0); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let b = wg.y;
  let x0 = i32(wg.x % P.tilesX) * 16;
  let y0 = i32(wg.x / P.tilesX) * 16;
  let lx = t % 16u;
  let ly = t / 16u;
  let C4 = P.C / 4u;
  ${g.map(u=>`var a${u} = 0.0;`).join(" ")}
  for (var c4 = 0u; c4 < C4; c4++) {
    for (var i = t; i < ${n*n}u; i += 256u) {
      ${d}
    }
    for (var i = t; i < ${a*r*r}u; i += 256u) { ws[i] = WT[i * C4 + c4]; }
    workgroupBarrier();
    for (var ky = 0u; ky < ${r}u; ky++) {
      for (var kx = 0u; kx < ${r}u; kx++) {
        let v = xs[(ly + ky) * ${n}u + lx + kx];
        let tap = ky * ${r}u + kx;
        ${g.map(u=>`a${u} += dot(v, ws[${u*r*r}u + tap]);`).join(" ")}
      }
    }
    workgroupBarrier();
  }
  let x = u32(x0) + lx;
  let y = u32(y0) + ly;
  if (x >= P.W || y >= P.H) { return; }
  let o = ((b * P.H + y) * P.W + x) * ${a}u;
  ${g.map(u=>`Y[o + ${u}u] = ${e(u)};`).join(`
  `)}
}`}),"convFewShader"),U=i(()=>t("firdown",()=>`
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
}`),"firDownShader"),F=i(r=>t(`styledact${r}`,()=>`
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
  if (c < P.C4) {${r?`
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
    }`:`
    v = X[pix * P.C4 + c];`}
    v = v * (inverseSqrt(D[b * P.C4 + c] + 1e-8) * 1.4142135) + ACT[c];
  } else {
    v = P.noise * F[pix * P.C4 + c - P.C4] + ACT[c];
  }
  Y[i] = select(v * 0.2, v, v >= vec4<f32>(0.0));
}`),"styledActShader"),G=i(r=>t(`torgb${r}`,()=>`
struct Params { n: u32, C4: u32, H: u32, W: u32, nx: u32, styOff: u32, styStride: u32, p0: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> STY: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WT: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> BIAS: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
${r?`@group(0) @binding(6) var<storage, read> SKIP: array<f32>;
fn sk(b: u32, yy: i32, xx: i32, ch: u32) -> f32 {
  let h = i32(P.H / 2u);
  let w = i32(P.W / 2u);
  if (yy < 0 || xx < 0 || yy >= h || xx >= w) { return 0.0; }
  return SKIP[((b * u32(h) + u32(yy)) * u32(w) + u32(xx)) * 3u + ch];
}`:""}
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
  var o = vec3<f32>(a0 + BIAS[0], a1 + BIAS[1], a2 + BIAS[2]);${r?`
  let x = p % P.W;
  let y = (p / P.W) % P.H;
  let y0 = (i32(y) - 1) >> 1u;       // floor((y - 1) / 2)
  let x0 = (i32(x) - 1) >> 1u;
  let wy = select(0.25, 0.75, (y & 1u) == 0u);  // weight of row y0 + 1
  let wx = select(0.25, 0.75, (x & 1u) == 0u);
  for (var ch = 0u; ch < 3u; ch++) {
    o[ch] += mix(mix(sk(b, y0, x0, ch), sk(b, y0, x0 + 1, ch), wx), mix(sk(b, y0 + 1, x0, ch), sk(b, y0 + 1, x0 + 1, ch), wx), wy);
  }`:""}
  OUT[p * 3u] = o.x;
  OUT[p * 3u + 1u] = o.y;
  OUT[p * 3u + 2u] = o.z;
}`),"toRgbShader"),M=i(()=>t("pixnorm",()=>`
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
}`),"pixelNormShader"),D=i(()=>t("square",()=>`
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = X[i] * X[i]; }
}`),"squareShader"),K=i(()=>t("banddown",()=>`
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
}`),"bandDownShader"),L=i(r=>t(`enhout${r}`,()=>`
struct Params { n: u32, N: u32, LO: u32, nx: u32, r: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;
${r?`@group(0) @binding(3) var<storage, read> D: array<f32>;
@group(0) @binding(4) var<storage, read> C: array<f32>;
fn dp(b: u32, x: i32, y: i32) -> vec3<f32> {
  let m = i32(P.LO) - 1;
  let i = ((b * P.LO + u32(clamp(y, 0, m))) * P.LO + u32(clamp(x, 0, m))) * 3u;
  return vec3<f32>(D[i], D[i + 1u], D[i + 2u]);
}`:""}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.y * P.nx * 256u + gid.x;
  if (p >= P.n) { return; }
  var v = vec3<f32>(G[p * 3u], G[p * 3u + 1u], G[p * 3u + 2u]);${r?`
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
  v = (v - vec3<f32>(C[p * 3u], C[p * 3u + 1u], C[p * 3u + 2u])) * 0.5;`:`
  v = v * 0.5 + 0.5;`}
  OUT[p * 3u] = v.x;
  OUT[p * 3u + 1u] = v.y;
  OUT[p * 3u + 2u] = v.z;
}`),"enhanceOutShader"),E=i(r=>t(`specrows${r}`,()=>`
struct Params { b0: u32, p1: u32, p2: u32, p3: u32 }; // b0: the first crop (0)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> F: array<f32>;
const N = ${r}u;
const TAU = 6.283185307179586;
var<workgroup> row: array<f32, N>;
fn hann(i: u32) -> f32 { return 0.5 - 0.5 * cos(TAU * f32(i) / f32(N - 1u)); }
@compute @workgroup_size(${r})
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
}`),"spectrumRowsShader"),V=i(r=>t(`speccols${r}`,()=>`
struct Params { b0: u32, p1: u32, p2: u32, p3: u32 }; // b0: the first crop (0)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read_write> PW: array<f32>;
const N = ${r}u;
const TAU = 6.283185307179586;
var<workgroup> col: array<vec2<f32>, N>;
@compute @workgroup_size(${r})
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
}`),"spectrumColsShader"),J=i(r=>t(`specrad${r}`,()=>`
struct Params { b0: u32, p1: u32, p2: u32, p3: u32 }; // b0: the first crop (0)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> PW: array<f32>;
@group(0) @binding(2) var<storage, read_write> R: array<f32>;
const N = ${r}u;
var<workgroup> acc: array<f32, N>;
@compute @workgroup_size(${r})
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
}`),"spectrumRadialShader"),Q=i(()=>t("frn",()=>`
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
}`),"frnShader"),Z=i(()=>t("frnparts",()=>`
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
}`),"frnPartsShader"),rr=i(()=>t("frncombine",()=>`
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
}`),"frnCombineShader"),er=i(({shuffle:r=!1,skip:a=!1,rotate:o=!1})=>t(`frnout${r}${a}${o}`,()=>`
struct Params { n4: u32, C4: u32, CS4: u32, OH: u32, OW: u32, B: u32, nx: u32, p0: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> PRO: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y: array<vec4<f32>>;
${a?"@group(0) @binding(4) var<storage, read> SKIP: array<vec4<f32>>;":""}
fn value(i: u32) -> vec4<f32> {
  let c = i % (P.C4 + P.CS4);
  let pix = i / (P.C4 + P.CS4);
  ${a?"if (c >= P.C4) { return SKIP[pix * P.CS4 + c - P.C4]; }":""}
  let b = pix / (P.OH * P.OW);
  ${r?`let x = pix % P.OW;
  let y = (pix / P.OW) % P.OH;
  let src = ((b * (P.OH / 2u) + y / 2u) * (P.OW / 2u) + x / 2u) * 4u * P.C4 + ((y & 1u) * 2u + (x & 1u)) * P.C4 + c;`:"let src = pix * P.C4 + c;"}
  let s = b * P.C4 + c;
  let BC4 = P.B * P.C4;
  return max(X[src] * PRO[s] + PRO[BC4 + s], PRO[2u * BC4 + s]);
}
${o?`var<workgroup> S: array<f32, 256>;
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
}`:`@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n4) { Y[i] = value(i); }
}`}`),"frnOutShader"),ir=i(r=>t(`blurdown${r}`,()=>{const a=r.reduce((n,s)=>n+s,0),o=i(n=>`${r[n]/a}`,"wt");return`
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
  var acc = vec4<f32>(0.0);${r.map((n,s)=>`
  { let sy = 2 * i32(y) + ${s-1};
    if (sy >= 0 && sy < i32(P.H)) {
      let r = (b * P.H + u32(sy)) * P.W;
      var row = vec4<f32>(0.0);${r.map((g,d)=>`
      { let sx = 2 * i32(x) + ${d-1}; if (sx >= 0 && sx < i32(P.W)) { row += X[(r + u32(sx)) * P.C4 + c] * ${o(d)}; } }`).join("")}
      acc += row * ${o(s)};
    } }`).join("")}
  Y[i] = acc;
}`}),"blurDownShader"),ar=i(()=>t("nchw",()=>`
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
}`),"nchwShader"),ur=i(r=>t(`maskmorph${r}`,()=>`
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
      v = ${r?"min":"max"}(v, X[base + yy * N + xx]);
    }
  }
  Y[i] = v;
}`),"maskMorphShader"),tr=i((r,a)=>t(`maskblur${r}${a}`,()=>`
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
  let pos = i32(${r?"(i / P.N) % P.N":"i % P.N"});
  let base = i - u32(pos) * ${r?"P.N":"1u"};
  var v = 0.0;
  for (var t = 0u; t <= 2u * P.r; t++) {
    var j = abs(pos + i32(t) - i32(P.r));
    if (j >= N) { j = 2 * (N - 1) - j; }
    v += X[base + u32(j) * ${r?"P.N":"1u"}] * TAPS[t];
  }
  Y[i] = ${a?"(clamp(v, 0.5, 1.0) - 0.5) * 2.0":"v"};
}`),"maskBlurShader"),or=i(()=>t("packhalf",()=>`
struct Params { n2: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n2) { Y[i] = pack2x16float(X[i]); }
}`),"packHalfShader"),nr=i(()=>t("zero",()=>`
struct Params { n: u32, nx: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * P.nx * 256u + gid.x;
  if (i < P.n) { Y[i] = 0.0; }
}`),"zeroShader");export{W as adainShader,N as affineShader,K as bandDownShader,ir as blurDownShader,I as convFewShader,Y as denseShader,L as enhanceOutShader,A as filterPrepShader,U as firDownShader,rr as frnCombineShader,er as frnOutShader,Z as frnPartsShader,Q as frnShader,h as hadamardShader,tr as maskBlurShader,ur as maskMorphShader,lr as matmulShader,ar as nchwShader,O as normCombineShader,H as normPartsShader,or as packHalfShader,z as pasteShader,M as pixelNormShader,R as pool2Shader,X as resampleShader,V as spectrumColsShader,J as spectrumRadialShader,E as spectrumRowsShader,S as splitReduceShader,D as squareShader,F as styledActShader,G as toRgbShader,j as unoccludeShader,B as upAddShader,T as winogradInShader,q as winogradOutShader,nr as zeroShader};
