// Tensor ops built on the WGSL kernels. Activations are f32 Tensors: images NHWC with shape
// [B, H, W, C], matrices [M, N]. Frames are { buf, w, h, stride }: RGBA8 pixels, one per u32.

import { grid } from "./device.js";
import * as K from "./kernels.js";

const mmParams = (o) => [
  ["u32", o.M], ["u32", o.N], ["u32", o.K], ["f32", o.alpha ?? 1],
  ["u32", o.lda ?? o.K], ["u32", o.aBatch ?? 0], ["u32", o.aOff ?? 0], ["u32", o.ldb ?? o.K],
  ["u32", o.bBatch ?? 0], ["u32", o.bDiv ?? 1], ["u32", o.bOff ?? 0], ["u32", o.ldc ?? o.N],
  ["u32", o.cBatch ?? 0], ["u32", o.cOff ?? 0], ["u32", o.gOff ?? 0], ["u32", o.cin ?? 0],
  ["u32", o.ih ?? 0], ["u32", o.iw ?? 0], ["u32", o.oh ?? 0], ["u32", o.ow ?? 0],
  ["u32", o.kw ?? 0], ["u32", o.stride ?? 1], ["u32", o.pad ?? 0], ["f32", o.slope ?? 0],
  ["u32", o.proOff ?? 0], ["u32", o.proStride ?? 0], ["u32", o.kChunk ?? 0], ["u32", o.phases ?? 0],
];

// A GEMM with fewer tiles than this is split along K (enough workgroups to fill a mid-size GPU)
const SPLIT_BELOW = 96;

function runMatmul(gpu, { a = "rows", A, W, C, bias, act = "none", resid = false, gate, add, prelu, pro, batch = 1, name, ...o }) {
  const b = W.kind;
  // big 128x128 tiles unless the problem is too small to fill them (or, with phases, a tile would
  // span two phases)
  const R = o.M >= 256 && o.N >= 128 && !(o.phases % 128) ? 8 : 4;
  const T = 16 * R;
  const TK = R === 8 ? 16 : 32;
  // vec4 loads need whole 8-wide k chunks and 4-aligned offsets/strides
  const al = (...xs) => xs.every((x) => (x ?? 0) % 4 === 0);
  const k8 = o.K % 8 === 0;
  const vecA = a === "rows" && k8 && al(o.lda ?? o.K, o.aOff, o.aBatch);
  const vecB = k8 && (b === "bf16" || b === "i8" || ((b === "f32" || b === "f16") && al(o.ldb ?? o.K, o.bOff, o.bBatch)));
  const kind = typeof a === "object" ? `conv${o.kw}x${o.kw}` : "gemm";
  const meta = { name: name || `${kind}.${b}`, flops: 2 * o.M * o.N * o.K * batch };

  // split-K: each split computes partial sums over at least 4 k-tiles
  const tiles = Math.ceil(o.N / T) * Math.ceil(o.M / T) * batch;
  const kTiles = Math.ceil(o.K / TK);
  let S = batch === 1 && !gate && tiles < SPLIT_BELOW ? Math.min(Math.ceil((2 * SPLIT_BELOW) / tiles), Math.floor(kTiles / 4)) : 1;
  if (S > 1) {
    const kChunk = Math.ceil(kTiles / S) * TK;
    S = Math.ceil(o.K / kChunk);
    const part = gpu.empty([S, o.M, o.N]);
    const code = K.matmulShader({ a, b, act, R, vecA, vecB, split: true });
    const bufs = [A, W.buf, part];
    if (b === "w4") bufs.push(W.srel, W.codebook);
    if (pro) bufs.push(pro);
    gpu.dispatch(code, bufs, mmParams({ ...o, cBatch: o.M * o.N, kChunk, alpha: 1 }), [Math.ceil(o.N / T), Math.ceil(o.M / T), S], meta);
    const scaled = b === "i8" || b === "w4";
    const rbufs = [part, C];
    if (scaled) rbufs.push(W.scale);
    if (bias) rbufs.push(bias);
    if (add) rbufs.push(add);
    if (act === "prelu") rbufs.push(prelu);
    const n = o.M * o.N;
    const [nx, ny] = grid(Math.ceil(n / 256));
    gpu.dispatch(K.splitReduceShader({ scaled, bias: !!bias, add: !!add, act, resid }), rbufs,
      [["u32", o.M], ["u32", o.N], ["u32", S], ["f32", o.alpha ?? 1], ["u32", nx], ["f32", o.slope ?? 0], ["u32", o.ldc ?? o.N], ["u32", o.cOff ?? 0]],
      [nx, ny], { name: meta.name + ".splitk" });
    part.release();
    return;
  }

  const code = K.matmulShader({ a, b, bias: !!bias, act, resid, gate: !!gate, R, vecA, vecB, add: !!add, phases: !!o.phases });
  const bufs = [A, W.buf, C];
  if (b === "i8" || b === "w4") bufs.push(W.scale);
  if (b === "w4") bufs.push(W.srel, W.codebook);
  if (bias) bufs.push(bias);
  if (gate) bufs.push(gate);
  if (add) bufs.push(add);
  if (act === "prelu") bufs.push(prelu);
  if (pro) bufs.push(pro);
  gpu.dispatch(code, bufs, mmParams(o), [Math.ceil(o.N / T), Math.ceil(o.M / T), batch], meta);
}

export const needsRotation = (W) => W.kind === "i8" || W.kind === "w4";

// ConvRot input rotation (Hadamard over each group of 256 features).
export function rotate(gpu, x) {
  const y = gpu.empty(x.shape);
  const groups = x.size / 256;
  const [nx, ny] = grid(groups);
  gpu.dispatch(K.hadamardShader(), [x, y], [["u32", groups], ["u32", nx]], [nx, ny], { name: "hadamard" });
  return y;
}

// y = x @ W^T (+ bias) (act, slope), x [M, K], or with { M, lda, aOff } the K columns from aOff of
// M rows lda floats apart. `xr` is the pre-rotated input for quantized weights; when omitted it
// is computed here (whole rows only).
export function linear(gpu, x, W, opts = {}) {
  const M = opts.M ?? x.size / W.K;
  let input = x;
  let tmp = null;
  if (needsRotation(W)) {
    if (opts.lda && !opts.xr) throw new Error(`${W.name}: a quantized linear on a column slice needs a rotated input`);
    input = opts.xr || (tmp = rotate(gpu, x));
  }
  const out = opts.out || gpu.empty([M, W.N]);
  runMatmul(gpu, { A: input, W, C: out, M, N: W.N, K: W.K, lda: opts.lda, aOff: opts.aOff, bias: opts.bias || W.bias, act: opts.act, slope: opts.slope, name: opts.name });
  tmp?.release();
  return out;
}

// 2D convolution of x [B, H, W, Cin] with a k x k weight [Cout, k*k*Cin] -> [B, OH, OW, Cout].
//  pad: before each axis, and after it unless padHi is given (zero padding only)
//  padMode: "zeros" | "reflect"; up: "" | "nearest" | "bilinear" (input upsampled 2x first)
//  pro: { buf, off } from adain() (or BN scale/shift), applied to the input as x*S + T
//       (proRelu: then max(., 0)); proStride: Cin per image, 0 to share one row.
//       { buf, off, scale: true }: x*S only, S of image b at off + b*proStride (modulated convs)
//  act: "none" | "relu" | "leaky" (slope) | "prelu" (per-channel slopes) | "tanh01"
//  add: a tensor of the output's shape added before the activation (residual)
//  phases: a transposed conv as a polyphase 2x2 conv (snake tap order, gemm.js) to 4 phases of
//          this many channels: zero taps are skipped
// Quantized weights take a ConvRot-rotated input: rotated here unless `xr` is given. A prologue
// can't be rotated on the fly, so pro and quantized weights don't mix (callers materialize).
export function conv2d(gpu, x, W, { k, stride = 1, pad = 0, padHi = pad, padMode = "zeros", up = "", act = "none", slope = 0.2, prelu = null, pro = null, proRelu = false, proStride, add = null, xr = null, phases = 0, name } = {}) {
  if (W.kind === "wino") {
    if (k !== 3 || stride !== 1 || pad !== 1 || padHi !== 1 || add || act === "prelu") throw new Error(`conv ${W.name}: unsupported by the Winograd path`);
    return conv2dWinograd(gpu, x, W, { padMode, up, act, slope, pro, proRelu, proStride, name });
  }
  const [B, H, Wd, cin] = x.shape;
  if (W.K !== k * k * cin) throw new Error(`conv ${W.name}: K ${W.K} != ${k}x${k}x${cin}`);
  if (padHi !== pad && padMode !== "zeros") throw new Error(`conv ${W.name}: uneven padding needs zero padding`);
  const LH = up ? 2 * H : H;
  const LW = up ? 2 * Wd : Wd;
  const oh = Math.floor((LH + pad + padHi - k) / stride) + 1;
  const ow = Math.floor((LW + pad + padHi - k) / stride) + 1;
  let input = x;
  let tmp = null;
  if (needsRotation(W)) {
    if (pro) throw new Error(`conv ${W.name}: a quantized conv needs a materialized input`);
    input = xr || (tmp = rotate(gpu, x));
  }
  const out = gpu.empty([B, oh, ow, W.N]);
  const a = { conv: true, vec: cin % 8 === 0, pad: padMode, up, pro: pro ? (pro.scale ? "scale" : proRelu ? "affine_relu" : "affine") : "", ...(phases ? { snake: true } : {}) };
  runMatmul(gpu, {
    a, A: input, W, C: out, M: B * oh * ow, N: W.N, K: W.K,
    cin, ih: H, iw: Wd, oh, ow, kw: k, stride, pad, slope,
    bias: W.bias, act, prelu, add, pro: pro?.buf, proOff: pro?.off, proStride: proStride ?? cin, phases, name,
  });
  tmp?.release();
  return out;
}

const filterBuffer = (gpu, bytes) => gpu.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

// f16 filters of a loaded 3x3 conv weight (any format, ConvRot undone) into a new buffer (or
// `buf`): out "winograd" -> [16][N][C] transformed, "ohwi" -> [N][9][C] dense.
function prepareFilters(gpu, W, out, buf = null) {
  const C = W.K / 9;
  const gs = C % 256 === 0 ? 256 : C;
  if (gs % 8 || gs > 256) throw new Error(`conv ${W.name}: ${C} input channels don't fit the filter preparation`);
  const rotated = W.kind === "i8" || W.kind === "w4";
  const vec = W.kind === "bf16" || W.kind === "f16" || W.kind === "i8" || W.kind === "f32";
  const halves = (out === "winograd" ? 16 : 9) * W.N * C;
  buf = buf || filterBuffer(gpu, halves * 2);
  const bufs = [W.buf];
  if (rotated) bufs.push(W.scale);
  if (W.kind === "w4") bufs.push(W.srel, W.codebook);
  bufs.push(buf);
  gpu.dispatch(K.filterPrepShader({ b: W.kind, vec, rotated, gs, out }), bufs,
    [["u32", W.N], ["u32", W.K], ["u32", W.K], ["u32", 0], ["u32", 0], ["u32", 1], ["u32", C], ["u32", 0]],
    [C / gs, W.N], { name: "filters." + out });
  return { buf, C, bytes: halves * 2 };
}

// Winograd filters from a loaded 3x3 conv weight (any format) -> { kind: "wino", N, C, buf, bias },
// and the source weight buffers are destroyed (the bias is kept). The transformed filters take
// ~3.5x the memory of int8 weights: when the GPU can't allocate them, returns null and W stays
// usable as a direct conv.
export async function winogradFilters(gpu, W) {
  const C = W.K / 9;
  gpu.device.pushErrorScope("out-of-memory");
  const target = filterBuffer(gpu, 16 * W.N * C * 2);
  if (await gpu.device.popErrorScope()) {
    target.destroy();
    return null;
  }
  const { buf } = prepareFilters(gpu, W, "winograd", target);
  gpu.flush();
  for (const k of ["buf", "scale", "srel", "codebook"]) W[k]?.destroy(); // destroy waits for the submitted work
  return { name: W.name, kind: "wino", N: W.N, C, buf, bias: W.bias };
}

// The dense f16 filter [N][9][C] (OHWI, rotation undone) of a loaded 3x3 conv weight, read back
// to the CPU as a Uint16Array (raw f16 bits).
export async function denseFiltersF16(gpu, W) {
  const { buf, bytes } = prepareFilters(gpu, W, "ohwi");
  gpu.flush();
  const staging = gpu.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpu.encode((enc) => enc.copyBufferToBuffer(buf, 0, staging, 0, bytes));
  await staging.mapAsync(GPUMapMode.READ);
  gpu.checkErrors();
  const out = new Uint16Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  buf.destroy();
  return out;
}

// The dense f16 matrix [N][K] (rotation undone) of a loaded matmul / conv weight in any format,
// read back to the CPU as a Uint16Array (raw f16 bits).
export async function denseWeightsF16(gpu, W) {
  if (W.K % 8) throw new Error(`${W.name}: K ${W.K} isn't a multiple of 8`);
  const rotated = W.kind === "i8" || W.kind === "w4";
  const bytes = W.N * W.K * 2;
  const buf = filterBuffer(gpu, bytes);
  const bufs = [W.buf];
  if (rotated) bufs.push(W.scale);
  if (W.kind === "w4") bufs.push(W.srel, W.codebook);
  bufs.push(buf);
  gpu.dispatch(K.denseShader({ b: W.kind, vec: W.kind !== "w4", rotated }), bufs,
    [["u32", W.N], ["u32", W.K], ["u32", W.K], ["u32", 0], ["u32", 0], ["u32", 1], ["u32", 0], ["u32", 0]],
    [Math.ceil(W.K / 256), W.N], { name: "dense.f16" });
  gpu.flush();
  const staging = gpu.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpu.encode((enc) => enc.copyBufferToBuffer(buf, 0, staging, 0, bytes));
  await staging.mapAsync(GPUMapMode.READ);
  gpu.checkErrors();
  const out = new Uint16Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  buf.destroy();
  return out;
}

// k x k "same" conv to a few output channels: x [B, H, W, C] with W = { k, N, wt (f32
// [N][k*k][C] buffer), bias } -> [B, H, W, N]. pad "reflect" | "zeros"; act "tanh01"
// ((tanh + 1) / 2) | "none".
export function convFew(gpu, x, W, name, { pad = "reflect", act = "tanh01" } = {}) {
  const [B, H, Wd, C] = x.shape;
  const y = gpu.empty([B, H, Wd, W.N]);
  const tilesX = Math.ceil(Wd / 16);
  gpu.dispatch(K.convFewShader(W.k, W.N, { pad, act }), [x, W.wt, W.bias, y], [["u32", H], ["u32", Wd], ["u32", C], ["u32", tilesX]],
    [tilesX * Math.ceil(H / 16), B], { name, flops: 2 * B * H * Wd * W.N * W.k * W.k * C });
  return y;
}

// Temporary Winograd buffers (V, M) stay below this size; larger convs run in chunks of tile rows
const WINO_BUDGET = 128 * 2 ** 20;

// Stride-1, pad-1 3x3 conv through Winograd F(2x2, 3x3) with W = { kind: "wino", N, C, buf
// (f16 [16][N][C]), bias }: input transform (with the direct path's pad / upsample / prologue
// options), 16 batched GEMMs, output transform (+bias, act).
function conv2dWinograd(gpu, x, W, { padMode, up, act, slope, pro, proRelu, proStride, name }) {
  const [B, H, Wd, C] = x.shape;
  if (C !== W.C) throw new Error(`conv ${W.name}: ${C} input channels, expected ${W.C}`);
  const N = W.N;
  const OH = up ? 2 * H : H;
  const OW = up ? 2 * Wd : Wd;
  const th = Math.ceil(OH / 2);
  const tw = Math.ceil(OW / 2);
  const tiles = B * th * tw;
  const budget = Math.min(gpu.maxBinding, WINO_BUDGET) / 4;
  // chunks of whole tile rows, so a large image can span several
  const chunk = Math.min(tiles, Math.max(tw, Math.floor(budget / (16 * Math.max(C, N)) / tw) * tw));
  const out = gpu.empty([B, OH, OW, N]);
  const pmode = pro ? (pro.scale ? "scale" : proRelu ? "affine_relu" : "affine") : "";
  for (let t0 = 0; t0 < tiles; t0 += chunk) {
    const T = Math.min(chunk, tiles - t0);
    const V = gpu.empty([16, T, C]);
    let [nx, ny] = grid(Math.ceil((T * C) / 4 / 64));
    gpu.dispatch(K.winogradInShader({ pad: padMode, up, pro: pmode }), pro ? [x, V, pro.buf] : [x, V],
      [["u32", H], ["u32", Wd], ["u32", C], ["u32", th], ["u32", tw], ["u32", T], ["u32", t0], ["u32", nx], ["u32", pro?.off ?? 0], ["u32", proStride ?? C], ["u32", 0], ["u32", 0]],
      [nx, ny], { name: (name || "conv") + ".wino-in" });
    const M = gpu.empty([16, T, N]);
    runMatmul(gpu, {
      A: V, W: { kind: "f16", buf: W.buf, N, K: C }, C: M, batch: 16, M: T, N, K: C,
      lda: C, aBatch: T * C, ldb: C, bBatch: N * C, ldc: N, cBatch: T * N, name: (name || "conv") + ".wino",
    });
    V.release();
    [nx, ny] = grid(Math.ceil((T * N) / 4 / 64));
    gpu.dispatch(K.winogradOutShader({ act, bias: !!W.bias }), W.bias ? [M, out, W.bias] : [M, out],
      [["u32", OH], ["u32", OW], ["u32", N], ["u32", th], ["u32", tw], ["u32", T], ["u32", t0], ["u32", nx], ["f32", slope], ["u32", 0], ["u32", 0], ["u32", 0]],
      [nx, ny], { name: (name || "conv") + ".wino-out" });
    M.release();
  }
  return out;
}

// Instance-norm statistics of h [B, H, W, C] combined with AdaIN style rows (sty: [scale C,
// shift C] per image, styleStride floats apart; 0 = one row for all) into the affine
// { buf, off } that conv2d / affine apply. A [ones, zeros] row gives the plain instance norm.
// Images larger than this many pixels are reduced in chunks of NORM_CHUNK pixels (a workgroup
// per 64 channels alone would walk the whole image)
const NORM_SPLIT = 1024;
const NORM_CHUNK = 512;

export function adain(gpu, h, sty, styleStride = 0, eps = 1e-8) {
  const [B, H, W, C] = h.shape;
  const HW = H * W;
  const buf = gpu.empty([2 * B * C]);
  if (HW <= NORM_SPLIT) {
    gpu.dispatch(K.adainShader(), [h, sty, buf], [["u32", HW], ["u32", C], ["u32", B], ["u32", styleStride], ["f32", eps]], [C / 64, B], { name: "adain" });
  } else {
    const chunks = Math.ceil(HW / NORM_CHUNK);
    const part = gpu.empty([B * chunks * 2 * C]);
    gpu.dispatch(K.normPartsShader(), [h, part], [["u32", HW], ["u32", C], ["u32", NORM_CHUNK], ["u32", chunks]], [C / 64, chunks, B], { name: "adain.parts" });
    gpu.dispatch(K.normCombineShader(), [part, sty, buf],
      [["u32", HW], ["u32", C], ["u32", B], ["u32", NORM_CHUNK], ["u32", chunks], ["u32", styleStride], ["f32", eps], ["u32", 0]],
      [Math.ceil((B * C) / 4 / 64)], { name: "adain.combine" });
    part.release();
  }
  return { buf, off: B * C, release: () => buf.release() };
}

// The [1,3,3,1] blur in front of the enhancer's stride-2 encoder convs: [B, H, W, C] -> [B, H+1, W+1, C].
export function firDown(gpu, x) {
  const [B, H, W, C] = x.shape;
  const y = gpu.empty([B, H + 1, W + 1, C]);
  const n4 = y.size / 4;
  const [nx, ny] = grid(Math.ceil(n4 / 256));
  gpu.dispatch(K.firDownShader(), [x, y], [["u32", n4], ["u32", C / 4], ["u32", H], ["u32", W], ["u32", nx], ["u32", 0], ["u32", 0], ["u32", 0]], [nx, ny], { name: "enhancer.blur" });
  return y;
}

// An enhancer modulated conv's activation (kernels.js styledActShader): y the conv output [B, H, W, C]
// (up: the polyphase transposed conv [B, H/2 + 1, W/2 + 1, 4C]), demod [B, C] (s^2 @ demod),
// act [2C], noise (the feature map's scalar), feat [B, H, W, C] -> [B, H, W, 2C].
export function styledAct(gpu, y, { demod, act, noise, feat, up = false }) {
  const [B, H, W, C] = feat.shape;
  const out = gpu.empty([B, H, W, 2 * C]);
  const n4 = out.size / 4;
  const [nx, ny] = grid(Math.ceil(n4 / 256));
  gpu.dispatch(K.styledActShader(up), [y, demod, act, feat, out],
    [["u32", n4], ["u32", C / 4], ["u32", H], ["u32", W], ["u32", nx], ["f32", noise], ["u32", 0], ["u32", 0]], [nx, ny], { name: up ? "enhancer.act-up" : "enhancer.act" });
  return out;
}

// the enhancer's modulated 1x1 conv to RGB (kernels.js toRgbShader): x [B, H, W, C], the style of image
// b at styles[off + b*stride ..], wt f32 [3*C], bias [3], skip [B, H/2, W/2, 3] or null ->
// [B, H, W, 3].
export function toRgb(gpu, x, { styles, off, stride, wt, bias, skip = null }) {
  const [B, H, W, C] = x.shape;
  const y = gpu.empty([B, H, W, 3]);
  const n = B * H * W;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.toRgbShader(!!skip), skip ? [x, styles, wt, bias, y, skip] : [x, styles, wt, bias, y],
    [["u32", n], ["u32", C / 4], ["u32", H], ["u32", W], ["u32", nx], ["u32", off / 4], ["u32", stride / 4], ["u32", 0]], [nx, ny],
    { name: "enhancer.rgb", flops: 2 * n * 3 * C });
  return y;
}

// Pixel norm of each row of x [rows, C]
export function pixelNorm(gpu, x) {
  const [rows, C] = x.shape;
  const y = gpu.empty(x.shape);
  gpu.dispatch(K.pixelNormShader(), [x, y], [["u32", rows], ["u32", C]], [Math.ceil(rows / 64)], { name: "pixelnorm" });
  return y;
}

// x * x, elementwise
export function square(gpu, x) {
  const y = gpu.empty(x.shape);
  const [nx, ny] = grid(Math.ceil(x.size / 256));
  gpu.dispatch(K.squareShader(), [x, y], [["u32", x.size], ["u32", nx]], [nx, ny], { name: "square" });
  return y;
}

// The part of c - g ([B, N, N, 3] each) below lo px, as a [B, lo, lo, 3] image (kernels.js
// bandDownShader).
export function bandDown(gpu, c, g, lo) {
  const [B, N] = c.shape;
  const d = gpu.empty([B, lo, lo, 3]);
  const s = N / lo;
  gpu.dispatch(K.bandDownShader(), [c, g, d], [["u32", N], ["u32", lo], ["u32", supersamples([s, 0, 0, 0, s, 0])], ["u32", 0], ["f32", s], ["u32", 0], ["u32", 0], ["u32", 0]],
    [Math.ceil(lo / 16), Math.ceil(lo / 16), B], { name: "band.down" });
  return d;
}

// The enhancer's output g ([B, N, N, 3] in [-1, 1]) for the paste, in [0, 1]; with band (from
// bandDown) the detail band of the input c (the enhancer's input) added back, as the change to
// c for an add paste: (g + up(band) - c) / 2.
export function enhanceOut(gpu, g, band = null, c = null) {
  const [B, N] = g.shape;
  const y = gpu.empty(g.shape);
  const n = B * N * N;
  const [nx, ny] = grid(Math.ceil(n / 256));
  const lo = band ? band.shape[1] : N;
  gpu.dispatch(K.enhanceOutShader(!!band), band ? [g, y, band, c] : [g, y],
    [["u32", n], ["u32", N], ["u32", lo], ["u32", nx], ["f32", lo / N], ["u32", 0], ["u32", 0], ["u32", 0]], [nx, ny], { name: "enhance.out" });
  return y;
}

// The radial power spectra of B face crops x [B, N, N, 3] (values 0..1, N = 256) -> [B, N/2 + 1]:
// per crop, the power of its Hann-windowed luma summed per radial frequency (kernels.js
// spectrum*Shader).
export function faceSpectrum(gpu, x) {
  const [B, N] = x.shape;
  const f = gpu.empty([B, N, N, 2]);
  gpu.dispatch(K.spectrumRowsShader(N), [x, f], [["u32", 0], ["u32", 0], ["u32", 0], ["u32", 0]], [N, B], { name: "spectrum.rows" });
  const pw = gpu.empty([B, N, N]);
  gpu.dispatch(K.spectrumColsShader(N), [f, pw], [["u32", 0], ["u32", 0], ["u32", 0], ["u32", 0]], [N, B], { name: "spectrum.cols" });
  f.release();
  const r = gpu.empty([B, N / 2 + 1]);
  gpu.dispatch(K.spectrumRadialShader(N), [pw, r], [["u32", 0], ["u32", 0], ["u32", 0], ["u32", 0]], [N / 2 + 1, B], { name: "spectrum.radial" });
  pw.release();
  return r;
}

// XSeg's FRN statistics of a conv's output x [B, H, W, groups*C] (groups 4: a polyphase transposed
// conv, whose 4 phases are the same channels), frn: the layer's [gamma, beta, tau] (3C), eps its
// epsilon -> pro { buf: [S, T, U] (B*C each), off: B*C } for frnOut.
export function frn(gpu, x, frnBuf, eps, groups = 1) {
  const [B, H, W, GC] = x.shape;
  const C = GC / groups;
  const HW = H * W;
  const buf = gpu.empty([3 * B * C]);
  if (HW <= NORM_SPLIT) {
    gpu.dispatch(K.frnShader(), [x, frnBuf, buf], [["u32", HW], ["u32", C], ["u32", B], ["u32", groups], ["f32", eps], ["u32", 0], ["u32", 0], ["u32", 0]],
      [Math.ceil(C / 64), B], { name: "frn" });
  } else {
    const chunks = Math.ceil(HW / NORM_CHUNK);
    const part = gpu.empty([B * chunks * GC]);
    gpu.dispatch(K.frnPartsShader(), [x, part], [["u32", HW], ["u32", GC], ["u32", NORM_CHUNK], ["u32", chunks]], [Math.ceil(GC / 64), chunks, B], { name: "frn.parts" });
    gpu.dispatch(K.frnCombineShader(), [part, frnBuf, buf],
      [["u32", HW], ["u32", C], ["u32", B], ["u32", groups], ["u32", chunks], ["f32", eps], ["u32", 0], ["u32", 0]],
      [Math.ceil((B * C) / 4 / 64)], { name: "frn.combine" });
    part.release();
  }
  return { buf, off: B * C, release: () => buf.release() };
}

// FRN + TLU applied to a conv's output x with pro from frn() (kernels.js frnOutShader) -> a new
// tensor. shuffle: x is a polyphase transposed conv [B, H, W, 4C] -> [B, 2H, 2W, C]; skip: a tensor
// of the output's size whose channels are appended; rotate: the result is ConvRot-rotated.
export function frnOut(gpu, x, pro, { shuffle = false, skip = null, rotate: rot = false } = {}) {
  const [B, H, W, XC] = x.shape;
  const C = shuffle ? XC / 4 : XC;
  const [OH, OW] = shuffle ? [2 * H, 2 * W] : [H, W];
  const CS = skip ? skip.shape[3] : 0;
  if (rot && (C + CS) % 256) throw new Error(`frnOut: can't rotate ${C + CS} channels`);
  const y = gpu.empty([B, OH, OW, C + CS]);
  const n4 = y.size / 4;
  const [nx, ny] = grid(rot ? n4 / 64 : Math.ceil(n4 / 256));
  gpu.dispatch(K.frnOutShader({ shuffle, skip: !!skip, rotate: rot }), skip ? [x, pro.buf, y, skip] : [x, pro.buf, y],
    [["u32", n4], ["u32", C / 4], ["u32", CS / 4], ["u32", OH], ["u32", OW], ["u32", B], ["u32", nx], ["u32", 0]], [nx, ny],
    { name: `frn.out${shuffle ? "+shuffle" : ""}${rot ? "+rot" : ""}` });
  return y;
}

// XSeg's blurred stride-2 downsampling (kernels.js blurDownShader): [B, H, W, C] -> [B, H/2, W/2, C]
export function blurDown(gpu, x, taps) {
  const [B, H, W, C] = x.shape;
  const y = gpu.empty([B, H / 2, W / 2, C]);
  const n4 = y.size / 4;
  const [nx, ny] = grid(Math.ceil(n4 / 256));
  gpu.dispatch(K.blurDownShader(taps), [x, y], [["u32", n4], ["u32", C / 4], ["u32", H], ["u32", W], ["u32", nx], ["u32", 0], ["u32", 0], ["u32", 0]], [nx, ny], { name: "xseg.blur" });
  return y;
}

// [B, H, W, C] -> its NCHW flatten [B, C*H*W]
export function flattenNCHW(gpu, x) {
  const [B, H, W, C] = x.shape;
  const y = gpu.empty([B, C * H * W]);
  const [nx, ny] = grid(Math.ceil(y.size / 256));
  gpu.dispatch(K.nchwShader(), [x, y], [["u32", y.size], ["u32", H * W], ["u32", C], ["u32", nx]], [nx, ny], { name: "nchw" });
  return y;
}

// the occlusion mask from XSeg's output m [B, N, N, 1]: grown by grow px of its crop (shrunk when
// negative), a separable Gaussian blur with taps (an uploaded buffer of 2r + 1 floats), then
// (clamp(m, 0.5, 1) - 0.5) * 2 -> [B, N, N, 1]
export function occlusionMask(gpu, m, taps, r, grow = 0) {
  const [, N] = m.shape;
  const n = m.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  let src = m;
  if (grow) {
    src = gpu.empty(m.shape);
    gpu.dispatch(K.maskMorphShader(grow < 0), [m, src], [["u32", n], ["u32", N], ["u32", Math.abs(grow)], ["u32", nx]], [nx, ny], { name: "mask.morph" });
  }
  const h = gpu.empty(m.shape);
  gpu.dispatch(K.maskBlurShader(false, false), [src, taps, h], [["u32", n], ["u32", N], ["u32", r], ["u32", nx]], [nx, ny], { name: "mask.blur" });
  if (grow) src.release();
  const y = gpu.empty(m.shape);
  gpu.dispatch(K.maskBlurShader(true, true), [h, taps, y], [["u32", n], ["u32", N], ["u32", r], ["u32", nx]], [nx, ny], { name: "mask.blur" });
  h.release();
  return y;
}

// Per-(image, channel) affine with pro = { buf, off } (proStride: C, or 0 for one shared row).
//  mode "relu" | "plain": returns a new tensor; mode "resid": out += x*S + T (in place);
//  mode "scale": x*S with S of image b at pro.off + b*proStride
//  rotate: the result is ConvRot-rotated (input of a quantized conv)
export function affine(gpu, x, pro, { mode = "plain", rotate = false, out = null, proStride } = {}) {
  const [B, H, W, C] = x.shape;
  const y = out || gpu.empty(x.shape);
  const rows = B * H * W;
  const [nx, ny] = grid(rows);
  gpu.dispatch(K.affineShader(mode, rotate), [x, pro.buf, y],
    [["u32", rows], ["u32", C], ["u32", H * W], ["u32", proStride ?? C], ["u32", pro.off], ["u32", nx]], [nx, ny],
    { name: `affine.${mode}${rotate ? "+rot" : ""}` });
  return y;
}

// 2x2 stride-2 pooling ("max" | "avg") of [B, H, W, C] with even H, W.
export function pool2(gpu, x, op) {
  const [B, H, W, C] = x.shape;
  const y = gpu.empty([B, H / 2, W / 2, C]);
  const n = y.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.pool2Shader(op), [x, y], [["u32", n], ["u32", C], ["u32", W / 2], ["u32", H / 2], ["u32", nx]], [nx, ny], { name: "pool." + op });
  return y;
}

// big += nearest-2x-upsample(small), in place
export function upAdd(gpu, small, big) {
  const [, H, W, C] = big.shape;
  const n = big.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.upAddShader(), [small, big], [["u32", n], ["u32", C], ["u32", W], ["u32", H], ["u32", nx]], [nx, ny], { name: "upadd" });
}

// The values of a tensor (even size) as f16 bits, read back to the CPU: half the bytes of
// gpu.read for data that is consumed in f16 anyway.
export async function readHalf(gpu, t) {
  const n2 = t.size / 2;
  const h = gpu.empty([n2]);
  const [nx, ny] = grid(Math.ceil(n2 / 256));
  gpu.dispatch(K.packHalfShader(), [t, h], [["u32", n2], ["u32", nx]], [nx, ny], { name: "pack.f16" });
  const data = await gpu.read(h);
  h.release();
  return new Uint16Array(data.buffer, data.byteOffset, t.size);
}

export function zero(gpu, t) {
  const n = t.size;
  const [nx, ny] = grid(Math.ceil(n / 256));
  gpu.dispatch(K.zeroShader(), [t], [["u32", n], ["u32", nx]], [nx, ny], { name: "zero" });
}

// Source pixels per output pixel of an affine map -> bilinear samples per axis (1..4).
export function supersamples(m) {
  const s = Math.sqrt(Math.abs(m[0] * m[4] - m[1] * m[3]));
  return Math.max(1, Math.min(4, Math.ceil(s - 1e-6)));
}

// Resamples a frame through m (2x3 row-major, output pixel -> frame pixel) into an RGB tensor:
// value * mul + add, `fill` at or beyond (vw, vh). k > 1: pixelshuffle layout [k*k, oh/k, ow/k, 3].
// out: a tensor or GPUBuffer to write into instead of a new tensor (then returned as is), from
// element `offset` on.
export function resample(gpu, frame, { ow, oh, m, mul = 1, add = 0, vw = ow, vh = oh, fill = 0, k = 1, zero: zeroBorder = false, out = null, offset = 0 }) {
  out = out || gpu.empty([k * k, oh / k, ow / k, 3]);
  gpu.dispatch(K.resampleShader(), [frame.buf, out], [
    ["u32", frame.w], ["u32", frame.h], ["u32", frame.stride], ["u32", ow],
    ["u32", oh], ["u32", vw], ["u32", vh], ["u32", k],
    ...m.map((v) => ["f32", v]), ["u32", supersamples(m)], ["u32", zeroBorder ? 1 : 0],
    ["f32", mul], ["f32", add], ["f32", fill], ["u32", offset],
  ], [Math.ceil(ow / 16), Math.ceil(oh / 16)], { name: "resample" });
  return out;
}

// Blends the swapped crop (pixelshuffle layout [k*k, S/k, S/k, 3], values 0..1) into the frame
// inside box [x1, y1, x2, y2), with m (2x3, frame pixel -> crop pixel) and the mask profile
// tensor [S] (the 2D mask is its outer product with itself), times alpha. add: swapped is a
// change to the frame (not clamped), added to it.
export function paste(gpu, frame, swapped, mask, m, box, S, k, alpha = 1, add = false) {
  const [x1, y1, x2, y2] = box;
  if (x2 <= x1 || y2 <= y1) return;
  gpu.dispatch(K.pasteShader(add), [frame.buf, swapped, mask], [
    ["u32", frame.stride], ["u32", x1], ["u32", y1], ["u32", x2],
    ["u32", y2], ["u32", S], ["u32", k], ["u32", S / k],
    ...m.map((v) => ["f32", v]), ["f32", alpha], ["u32", 0],
  ], [Math.ceil((x2 - x1) / 16), Math.ceil((y2 - y1) / 16)], { name: "paste" });
}

// Puts the original frame `orig` (a copy of the frame before any swap, same size) back into the
// frame inside box [x1, y1, x2, y2) where the occlusion mask occ ({ buf, off, m, size }: size x
// size floats from off, m its frame -> mask map; 1 outside it) is below 1.
export function unocclude(gpu, frame, orig, occ, box) {
  const [x1, y1, x2, y2] = box;
  if (x2 <= x1 || y2 <= y1) return;
  gpu.dispatch(K.unoccludeShader(), [frame.buf, orig.buf, occ.buf], [
    ["u32", frame.stride], ["u32", x1], ["u32", y1], ["u32", x2],
    ["u32", y2], ["u32", occ.size], ["u32", occ.off], ["u32", 0],
    ...occ.m.map((v) => ["f32", v]), ["u32", 0], ["u32", 0],
  ], [Math.ceil((x2 - x1) / 16), Math.ceil((y2 - y1) / 16)], { name: "unocclude" });
}
