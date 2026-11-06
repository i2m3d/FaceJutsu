var E=Object.defineProperty;var o=(t,u)=>E(t,"name",{value:u,configurable:!0});const c=o(t=>[...Array(t).keys()],"range"),K=new Map;function I(t){if(t)return`
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }
  let q = (P.aOff + z * P.aBatch + m * P.lda + k) >> 2u;
  return V8(A[q], A[q + 1u]);
}`;const u=c(8).map(r=>`select(0.0, A[base + ${r}u], k + ${r}u < P.K)`);return`
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }
  let base = P.aOff + z * P.aBatch + m * P.lda + k;  // out-of-range lanes are masked (robust access keeps reads safe)
  return V8(vec4<f32>(${u.slice(0,4).join(", ")}), vec4<f32>(${u.slice(4).join(", ")}));
}`}o(I,"loaderRows");function C({pad:t,up:u,snake:r}){const e=u?"2u * P.ih":"P.ih",i=u?"2u * P.iw":"P.iw",a=t==="reflect"?`iy = refl(iy, i32(${e})); ix = refl(ix, i32(${i}));`:`if (iy < 0 || ix < 0 || iy >= i32(${e}) || ix >= i32(${i})) { return false; }`;return`
fn refl(i: i32, n: i32) -> i32 { return select(select(i, 2 * n - 2 - i, i >= n), -i, i < 0); }
fn tapPos(oy: u32, ox: u32, stored: u32, pos: ptr<function, vec2<i32>>) -> bool {
  let tap = ${r?"select(stored, stored ^ 1u, stored >= 2u)":"stored"};
  var iy = i32(oy * P.stride + tap / P.kw) - i32(P.pad);
  var ix = i32(ox * P.stride + tap % P.kw) - i32(P.pad);
  ${a}
  *pos = vec2<i32>(iy, ix);
  return true;
}`}o(C,"convCoords");function M({vec:t,pad:u,up:r,pro:e,snake:i=!1}){const a=`
  let ox = m % P.ow;
  let t = m / P.ow;
  let oy = t % P.oh;
  let b = t / P.oh;`,x=e==="affine_relu";if(t){let l;r==="bilinear"?l=`
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
  var v = V8(mix(mix(a00.lo, a01.lo, wx), mix(a10.lo, a11.lo, wx), wy), mix(mix(a00.hi, a01.hi, wx), mix(a10.hi, a11.hi, wx), wy));`:l=`
  var v = ld8(((b * P.ih + ${r==="nearest"?"u32(p.x) / 2u":"u32(p.x)"}) * P.iw + ${r==="nearest"?"u32(p.y) / 2u":"u32(p.y)"}) * P.cin + ci);`;const f=e==="scale"?`
  let si = (P.proOff + b * P.proStride + ci) >> 2u;
  v = V8(v.lo * PRO[si], v.hi * PRO[si + 1u]);`:e?`
  let si = (b * P.proStride + ci) >> 2u;
  let so = si + (P.proOff >> 2u);
  v = V8(v.lo * PRO[si] + PRO[so], v.hi * PRO[si + 1u] + PRO[so + 1u]);${x?`
  v = V8(max(v.lo, vec4<f32>(0.0)), max(v.hi, vec4<f32>(0.0)));`:""}`:"";return`
fn ld8(i: u32) -> V8 { let q = i >> 2u; return V8(A[q], A[q + 1u]); }
${C({pad:u,up:r,snake:i})}
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M || k >= P.K) { return V8(); }${a}
  let tap = k / P.cin;
  let ci = k % P.cin;
  var p: vec2<i32>;
  if (!tapPos(oy, ox, tap, &p)) { return V8(); }${l}${f}
  return v;
}`}let v;r==="bilinear"?v=`
  let sy = max((f32(p.x) + 0.5) * 0.5 - 0.5, 0.0);
  let sx = max((f32(p.y) + 0.5) * 0.5 - 0.5, 0.0);
  let y0 = u32(sy);
  let x0 = u32(sx);
  let y1 = min(y0 + 1u, P.ih - 1u);
  let x1 = min(x0 + 1u, P.iw - 1u);
  let r0 = (b * P.ih + y0) * P.iw;
  let r1 = (b * P.ih + y1) * P.iw;
  var v = mix(mix(A[(r0 + x0) * P.cin + ci], A[(r0 + x1) * P.cin + ci], sx - f32(x0)),
              mix(A[(r1 + x0) * P.cin + ci], A[(r1 + x1) * P.cin + ci], sx - f32(x0)), sy - f32(y0));`:v=`
  var v = A[((b * P.ih + ${r==="nearest"?"u32(p.x) / 2u":"u32(p.x)"}) * P.iw + ${r==="nearest"?"u32(p.y) / 2u":"u32(p.y)"}) * P.cin + ci];`;const d=e==="scale"?`
  v = v * PRO[P.proOff + b * P.proStride + ci];`:e?`
  v = v * PRO[b * P.proStride + ci] + PRO[P.proOff + b * P.proStride + ci];${x?`
  v = max(v, 0.0);`:""}`:"";return`
${C({pad:u,up:r,snake:i})}
fn elemA(b: u32, oy: u32, ox: u32, k: u32) -> f32 {
  if (k >= P.K) { return 0.0; }
  let tap = k / P.cin;
  let ci = k % P.cin;
  var p: vec2<i32>;
  if (!tapPos(oy, ox, tap, &p)) { return 0.0; }${v}${d}
  return v;
}
fn loadA8(z: u32, m: u32, k: u32) -> V8 {
  if (m >= P.M) { return V8(); }${a}
  return V8(vec4<f32>(${c(4).map(l=>`elemA(b, oy, ox, k + ${l}u)`).join(", ")}),
            vec4<f32>(${c(4).map(l=>`elemA(b, oy, ox, k + ${l+4}u)`).join(", ")}));
}`}o(M,"loaderConv");function _(t,u){return t==="f32"?u?"vec4<f32>":"f32":t==="bf16"||t==="f16"?u?"vec4<u32>":"u32":t==="i8"&&u?"vec2<u32>":"u32"}o(_,"weightArrayType");function L(t,u){if(t==="f32"){if(u)return`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let q = (P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k) >> 2u;
  return V8(B[q], B[q + 1u]);
}`;const r=c(8).map(e=>`select(0.0, B[base + ${e}u], k + ${e}u < P.K)`);return`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N) { return V8(); }
  let base = P.bOff + (z / P.bDiv) * P.bBatch + n * P.ldb + k;
  return V8(vec4<f32>(${r.slice(0,4).join(", ")}), vec4<f32>(${r.slice(4).join(", ")}));
}`}if(t==="bf16"){const r=o(a=>`bitcast<f32>(${a} << 16u)`,"lo"),e=o(a=>`bitcast<f32>(${a} & 0xffff0000u)`,"hi");return u?`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let w = B[(n * P.K + k) >> 3u];
  return V8(vec4<f32>(${r("w.x")}, ${e("w.x")}, ${r("w.y")}, ${e("w.y")}), vec4<f32>(${r("w.z")}, ${e("w.z")}, ${r("w.w")}, ${e("w.w")}));
}`:`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let base = (n * P.K + k) >> 1u;
  ${c(4).map(a=>`let w${a} = select(0u, B[base + ${a}u], k + ${2*a}u < P.K);`).join(`
  `)}
  return V8(vec4<f32>(${r("w0")}, ${e("w0")}, ${r("w1")}, ${e("w1")}), vec4<f32>(${r("w2")}, ${e("w2")}, ${r("w3")}, ${e("w3")}));
}`}if(t==="f16")return u?`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let w = B[(P.bOff + z * P.bBatch + n * P.K + k) >> 3u];
  return V8(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)));
}`:`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  let base = (P.bOff + z * P.bBatch + n * P.K + k) >> 1u;
  ${c(4).map(e=>`let w${e} = select(vec2<f32>(), unpack2x16float(B[base + ${e}u]), k + ${2*e}u < P.K);`).join(`
  `)}
  return V8(vec4<f32>(w0, w1), vec4<f32>(w2, w3));
}`;if(t==="i8"){const r=u?`let w = B[(n * P.K + k) >> 3u];
  let w0 = i32(w.x);
  let w1 = i32(w.y);`:`let base = (n * P.K + k) >> 2u;
  let w0 = i32(B[base]);
  let w1 = i32(B[base + 1u]);`,e=o(i=>c(4).map(a=>`f32(extractBits(${i}, ${8*a}u, 8u))`).join(", "),"x");return`
fn loadB8(z: u32, n: u32, k: u32) -> V8 {
  if (n >= P.N || k >= P.K) { return V8(); }
  ${r}
  return V8(vec4<f32>(${e("w0")}), vec4<f32>(${e("w1")}));
}`}if(t==="w4"){const r=o(e=>`clamp(round(CB[(codes >> ${4*e}u) & 15u] * s), -127.0, 127.0)`,"lv");return`
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
  return V8(vec4<f32>(${c(4).map(r).join(", ")}), vec4<f32>(${c(4).map(e=>r(e+4)).join(", ")}));
}`}throw new Error(`unknown B format ${t}`)}o(L,"loaderB");function G({scaled:t=!1,bias:u=!1,add:r=!1,act:e="none"}){let i="";return t&&(i+=" r = r * RS[n];"),u&&(i+=" r = r + BIAS[n];"),r&&(i+=" r = r + ADD[m * P.N + n];"),e==="relu"&&(i+=" r = max(r, 0.0);"),e==="leaky"&&(i+=" r = select(r * P.slope, r, r >= 0.0);"),e==="prelu"&&(i+=" r = select(r * SL[n], r, r >= 0.0);"),e==="silu"&&(i+=" r = r / (1.0 + exp(-r));"),e==="tanh01"&&(i+=" r = 1.0 / (1.0 + exp(-2.0 * r));"),i}o(G,"epilogue");function F({a:t="rows",b:u="f32",bias:r=!1,act:e="none",resid:i=!1,gate:a=!1,R:x=8,vecA:v=!1,vecB:d=!1,add:l=!1,split:f=!1,phases:B=!1}){const y=JSON.stringify([t,u,r,e,i,a,x,v,d,l,f,B]);if(K.has(y))return K.get(y);const V=typeof t=="object",N=V?t.vec:v;f&&(r=l=a=i=!1);const O=!f&&(u==="i8"||u==="w4"),z=[],s=o(n=>z.push(`@group(0) @binding(${z.length+1}) ${n};`),"bind");s(`var<storage, read> A: array<${N?"vec4<f32>":"f32"}>`),s(`var<storage, read> B: array<${_(u,d)}>`),s("var<storage, read_write> C: array<f32>"),O&&s("var<storage, read> RS: array<f32>"),u==="w4"&&(s("var<storage, read> SR: array<u32>"),s("var<storage, read> CB: array<f32>")),r&&s("var<storage, read> BIAS: array<f32>"),a&&s("var<storage, read> G: array<f32>"),l&&s("var<storage, read> ADD: array<f32>"),e==="prelu"&&!f&&s("var<storage, read> SL: array<f32>"),V&&t.pro&&s(`var<storage, read> PRO: array<${t.vec?"vec4<f32>":"f32"}>`);const R=f?"":G({scaled:O,bias:r,add:l,act:e}),T=i?a?"C[ci] = C[ci] + G[P.gOff + n] * r;":"C[ci] = C[ci] + r;":"C[ci] = r;",q=f?"z * P.cBatch + m * P.N + n":"P.cOff + z * P.cBatch + m * P.ldc + n",j=f?"":" * P.alpha",w=16*x,h=x===8?16:32,p=w/4,P=x/4,A=h/8,D=c(8).map(n=>{const $=n<4?`lo.${"xyzw"[n]}`:`hi.${"xyzw"[n-4]}`;return`As[(lk + ${n}u) * ${p}u + sIdx][sLane] = av.${$};
      Bs[(lk + ${n}u) * ${p}u + sIdx][sLane] = bv.${$};`}),b=[];for(let n=0;n<P;n++)b.push(`let a${n} = As[kk * ${p}u + ${n*16}u + tm];`);for(let n=0;n<P;n++)b.push(`let b${n} = Bs[kk * ${p}u + ${n*16}u + tn];`);for(let n=0;n<P;n++)["x","y","z","w"].forEach(($,m)=>{for(let k=0;k<P;k++)b.push(`acc${(n*4+m)*P+k} += a${n}.${$} * b${k};`)});const g=[];for(let n=0;n<P;n++)for(let $=0;$<4;$++){g.push(`{ let m = m0 + ${n*64+$}u + tm * 4u;
    if (m < P.M) {`);for(let m=0;m<P;m++)for(let k=0;k<4;k++)g.push(`      { let n = n0 + ${m*64+k}u + tn * 4u; if (n < P.N) { let ci = ${q}; var r = acc${(n*4+$)*P+m}.${"xyzw"[k]}${j};${R} ${T} } }`);g.push("    } }")}const S=`
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
${z.join(`
`)}
${V?M(t):I(v)}
${L(u,d)}

const TK = ${h}u;
var<workgroup> As: array<vec4<f32>, ${p*h}>;  // [TK][T/4]
var<workgroup> Bs: array<vec4<f32>, ${p*h}>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let z = wg.z;
  ${f?`let kBegin = z * P.kChunk;
  let kEnd = min(P.K, kBegin + P.kChunk);
  let zl = 0u; // the split index isn't a batch index`:B?`// the taps of the phases this workgroup's columns belong to
  let ph0 = (wg.x * ${w}u) / P.phases;
  let ph1 = min((wg.x * ${w}u + ${w-1}u) / P.phases, 3u);
  let kBegin = min(ph0, 2u) * P.cin;
  let kEnd = select(3u, 4u, ph0 == 0u || (ph0 <= 2u && ph1 >= 2u)) * P.cin;
  let zl = z;`:`let kBegin = 0u;
  let kEnd = P.K;
  let zl = z;`}
  let m0 = wg.y * ${w}u;
  let n0 = wg.x * ${w}u;
  let tm = t & 15u;
  let tn = t >> 4u;
  let lr = t / ${A}u;
  let lk = (t % ${A}u) * 8u;
  let sIdx = lr >> 2u;
  let sLane = lr & 3u;
  ${c(x*P).map(n=>`var acc${n} = vec4<f32>();`).join(`
  `)}
  var av = loadA8(zl, m0 + lr, kBegin + lk);
  var bv = loadB8(zl, n0 + lr, kBegin + lk);

  for (var k0 = kBegin; k0 < kEnd; k0 += TK) {
    ${D.join(`
    `)}
    workgroupBarrier();
    if (k0 + TK < kEnd) {
      av = loadA8(zl, m0 + lr, k0 + TK + lk);
      bv = loadB8(zl, n0 + lr, k0 + TK + lk);
    }
    for (var kk = 0u; kk < TK; kk++) {
      ${b.join(`
      `)}
    }
    workgroupBarrier();
  }

  ${g.join(`
  `)}
}`;return K.set(y,S),S}o(F,"matmulShader");export{G as epilogue,L as loaderB,F as matmulShader,_ as weightArrayType};
