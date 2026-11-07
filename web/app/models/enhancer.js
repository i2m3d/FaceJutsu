// The face enhancer, at 256 or 512 px: aligned RGB face crops
// [B, N, N, 3] in [-1, 1] -> restored crops in the same layout, unclamped.
//
//   encoder  1x1 conv, then per level the [1,3,3,1] blur and a stride-2 3x3 conv, down to 4 px
//   style    a linear on the 4 px map, pixel norm, 8 linears -> w; then every layer's style
//            s = linear(w) in one GEMM, and the demodulation sums s^2 @ demod^T per layer
//   decoder  from a learned 4 px input, per resolution: the stride-2 transposed conv, run as a
//            2x2 conv to 4x the channels (polyphase, expanded from the stored 3x3 kernel at
//            load), then a 3x3 conv. Both are modulated (the conv scales its input by s while
//            loading it) and demodulated in the activation kernel, which also blurs the
//            transposed conv and appends the encoder's feature map of that size. A 1x1 conv to
//            RGB per resolution, added to the upsampled image of the resolution before.
//
// Quantized direct convs need a ConvRot-rotated input, so there the scaled input is materialized
// together with the rotation. With `winograd` (the default) the 3x3 convs from 16 to 256 px run
// as Winograd F(2x2, 3x3) (inswapper.js): below that they are small and weight-bound, and at
// 512 px (128 -> 64 channels) the direct conv is as fast and needs no transform buffers.

import * as ops from "../gpu/ops.js";
import { polyphase } from "../weights.js";

const WINOGRAD = [16, 256]; // px

export class FaceEnhancer {
  static async load(gpu, st, onProgress, { winograd = true } = {}) {
    let levels = 0;
    while (st.has(`enc.${levels}.weight`)) levels++;
    const m = new FaceEnhancer(gpu, 2 ** (levels + 1));
    const names = [...Array(levels).keys()].map((i) => `enc.${i}`).concat("fc", ...[...Array(8).keys()].map((j) => `style.${j}`), "mod");
    // decoder layers in style order (build_enhancer): conv.0, rgb.0, then per resolution up.i,
    // conv.i+1, rgb.i+1; each with its style's offset in the row of all styles and its input
    // channel count
    const layers = [{ name: "conv.0", kind: "conv", res: 4, feat: levels - 1 }, { name: "rgb.0", kind: "rgb" }];
    for (let i = 0; i < levels - 1; i++) {
      const res = 8 * 2 ** i;
      layers.push({ name: `up.${i}`, kind: "up", res, feat: levels - 2 - i }, { name: `conv.${i + 1}`, kind: "conv", res, feat: levels - 2 - i }, { name: `rgb.${i + 1}`, kind: "rgb" });
    }
    const total = names.length + layers.length;
    let done = 0;
    for (const n of names) {
      m.w[n] = await st.linear(gpu, n);
      onProgress?.(++done / total);
    }
    const c = await st.f32("const");
    m.w.const = { buf: gpu.upload(c), bytes: c.byteLength, C: c.length / 16 };
    let off = 0;
    for (const L of layers) {
      if (L.kind === "rgb") {
        const wt = await st.f32(L.name + ".weight");
        Object.assign(L, { cin: wt.length / 3, wt: gpu.upload(wt), bias: await st.vector(gpu, L.name + ".bias") });
      } else {
        let W = L.kind === "up" ? await polyphase(gpu, st, L.name) : await st.linear(gpu, L.name);
        L.cin = L.kind === "up" ? W.K / 4 : W.K / 9;
        if (winograd && L.kind === "conv" && L.res >= WINOGRAD[0] && L.res <= WINOGRAD[1]) {
          // once the GPU runs out of memory for Winograd filters, the remaining convs stay direct
          const wino = await ops.winogradFilters(gpu, W);
          if (wino) W = wino;
          else {
            console.warn(`not enough GPU memory for Winograd filters from ${L.name} on: direct convolutions`);
            winograd = false;
          }
        }
        m.w[L.name] = W;
        Object.assign(L, {
          demod: await st.linear(gpu, L.name + ".demod"),
          act: await st.vector(gpu, L.name + ".act"),
          noise: (await st.f32(L.name + ".noise"))[0],
        });
      }
      L.off = off;
      off += L.cin;
      onProgress?.(++done / total);
    }
    if (off !== m.w.mod.N) throw new Error(`the enhancer: the layers' styles take ${off} of ${m.w.mod.N} values`);
    m.w.layers = layers;
    m.levels = levels;
    return m;
  }

  constructor(gpu, size) {
    this.gpu = gpu;
    this.w = {};
    this.levels = 0;
    this.crop = { size, mul: 2 / 255, add: -1 }; // input crops in [-1, 1]
  }

  // x: [B, N, N, 3] in [-1, 1] -> [B, N, N, 3]
  forward(x) {
    const gpu = this.gpu;
    const w = this.w;
    const B = x.shape[0];
    // the decoder's learned input, once per image (copied first: a copy submits the work so far)
    let h = gpu.empty([B, 4, 4, w.const.C]);
    gpu.encode((enc) => { for (let b = 0; b < B; b++) enc.copyBufferToBuffer(w.const.buf, 0, h.buf, b * w.const.bytes, w.const.bytes); });
    const lrelu = { act: "leaky", slope: 0.2 };
    const conv = (input, name, opts) => ops.conv2d(gpu, input, w[name], { name, ...opts });
    const step = (t, fn) => { const y = fn(t); t.release(); return y; };

    // encoder: feats[i] at N / 2^i px
    const feats = [conv(x, "enc.0", { k: 1, ...lrelu })];
    for (let i = 1; i < this.levels; i++) feats.push(step(ops.firDown(gpu, feats[i - 1]), (t) => conv(t, `enc.${i}`, { k: 3, stride: 2, ...lrelu })));

    // style: w, then every layer's style [B, S] and its square (for the demodulation)
    let z = ops.linear(gpu, feats[this.levels - 1], w.fc, { ...lrelu, name: "enhancer.fc" });
    z = step(z, (t) => ops.pixelNorm(gpu, t));
    for (let j = 0; j < 8; j++) z = step(z, (t) => ops.linear(gpu, t, w[`style.${j}`], { ...lrelu, name: "enhancer.style" }));
    const styles = step(z, (t) => ops.linear(gpu, t, w.mod, { name: "enhancer.mod" }));
    const S = w.mod.N;
    const sq = ops.square(gpu, styles);

    // a modulated conv's raw output: its input scaled by the layer's style while loading
    const modConv = (h, L) => {
      const pro = { buf: styles, off: L.off, scale: true };
      const shape = L.kind === "up" ? { k: 2, pad: 1, phases: w[L.name].N / 4 } : { k: 3, pad: 1 };
      if (!ops.needsRotation(w[L.name])) return conv(h, L.name, { ...shape, pro, proStride: S });
      const r = ops.affine(gpu, h, pro, { mode: "scale", rotate: true, proStride: S });
      const y = conv(r, L.name, { ...shape, xr: r });
      r.release();
      return y;
    };

    // decoder
    let rgb = null;
    for (const L of w.layers) {
      if (L.kind === "rgb") {
        const y = ops.toRgb(gpu, h, { styles, off: L.off, stride: S, wt: L.wt, bias: L.bias, skip: rgb });
        rgb?.release();
        rgb = y;
        continue;
      }
      const y = modConv(h, L);
      const d = ops.linear(gpu, sq, L.demod, { M: B, lda: S, aOff: L.off, name: "enhancer.demod" });
      const f = feats[L.feat];
      h = step(h, () => ops.styledAct(gpu, y, { demod: d, act: L.act, noise: L.noise, feat: f, up: L.kind === "up" }));
      y.release();
      d.release();
      // a feature map's last reader is the 3x3 conv of its resolution
      if (L.kind === "conv") f.release();
    }
    h.release();
    styles.release();
    sq.release();
    return rgb;
  }
}
