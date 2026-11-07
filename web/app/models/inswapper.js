// inswapper_128: aligned RGB face crops [B, 128, 128, 3] in [0, 1] and a 512-d
// identity latent -> swapped crops in the same layout.
//
//   encoder  7x7 conv (reflect pad), 3x3 conv, two stride-2 3x3 convs; LeakyReLU(0.2) -> 32x32x1024
//   6 blocks x += AdaIN2(conv2(relu(AdaIN1(conv1(x))))): 3x3 convs with reflect padding, instance
//            norm (eps 1e-8) whose scale/shift come from a linear on the latent
//   decoder  2x (bilinear 2x upsample + 3x3 conv), 3x3 conv, LeakyReLU; 7x7 conv, (tanh + 1) / 2
//
// Each AdaIN folds into one per-(image, channel) affine (ops.adain). For bf16/f32 convs the first
// one is applied while conv2 loads its input; quantized convs need a ConvRot-rotated input, so
// there it is materialized together with the rotation.
//
// With `winograd` (the default) the stride-1 3x3 convs, 94% of the work, run as Winograd
// F(2x2, 3x3): their filters are dequantized (rotation undone) and transformed on the GPU at load
// into f16 [16][Cout][Cin] (about 3x the memory of the int8 weights, 2.25x fewer multiplies).

import * as ops from "../gpu/ops.js";

const CONVS = [
  "enc.0", "enc.1", "enc.2", "enc.3",
  ...[0, 1, 2, 3, 4, 5].flatMap((i) => [`styles.${i}.conv1`, `styles.${i}.conv2`]),
  "dec.0", "dec.1", "dec.2",
];
const STYLES = [0, 1, 2, 3, 4, 5].flatMap((i) => [`styles.${i}.style1`, `styles.${i}.style2`]);
const WINOGRAD = new Set(["enc.1", "dec.0", "dec.1", "dec.2", ...CONVS.filter((n) => n.startsWith("styles."))]);

export class Inswapper {
  static async load(gpu, st, onProgress, { winograd = true } = {}) {
    const m = new Inswapper(gpu);
    const names = [...CONVS, ...STYLES];
    for (let i = 0; i < names.length; i++) {
      const n = names[i];
      const W = await st.linear(gpu, n);
      // once the GPU runs out of memory for Winograd filters, the remaining convs stay direct
      const wino = winograd && WINOGRAD.has(n) ? await ops.winogradFilters(gpu, W) : null;
      if (winograd && WINOGRAD.has(n) && !wino) {
        console.warn(`not enough GPU memory for Winograd filters from ${n} on: direct convolutions`);
        winograd = false;
      }
      m.w[n] = wino || W;
      onProgress?.((i + 1) / names.length);
    }
    // the last conv (7x7, 128 -> 3 channels) runs as a direct conv kernel with f32 weights
    m.w.out = { k: 7, N: 3, wt: gpu.upload(Float32Array.from(await st.f32("out.weight"))), bias: await st.vector(gpu, "out.bias") };
    m.emap = await st.f32("emap");
    return m;
  }

  constructor(gpu) {
    this.gpu = gpu;
    this.w = {};
    this.crop = { size: 128, mul: 1 / 255, add: 0 }; // input crops in [0, 1]
    // the detail a swap at pixelshuffle 1 to 4 really has, in px of the 128 px crop's framing
    // (tools/measure_detail.py): more pixels, but not as much more detail
    this.detail = [70, 108, 124, 128];
    // a face with less detail of its own (face.faceDetail) gives a swap with less:
    // softmin(a * its detail, detail at the pixelshuffle), pipeline swapDetail
    this.detailFit = { a: 0.944, p: 2.5 };
  }

  // ArcFace embedding -> identity latent: normalize, project
  // through emap, normalize again.
  latent(embedding) {
    const e = normalize(embedding);
    const out = new Float32Array(512);
    for (let i = 0; i < 512; i++) {
      const ei = e[i];
      if (ei === 0) continue;
      const row = i * 512;
      for (let j = 0; j < 512; j++) out[j] += ei * this.emap[row + j];
    }
    return normalize(out);
  }

  // latent -> the 12 AdaIN style rows, [1, 2048] each (scale 1024, shift 1024). Computed once per
  // source face.
  styles(latent) {
    const x = this.gpu.fromArray(latent, [1, 512]);
    const out = STYLES.map((n) => ops.linear(this.gpu, x, this.w[n], { name: "style" }));
    x.release();
    return out;
  }

  // x: [B, 128, 128, 3]; styles: from styles(), shared by every image of the batch
  forward(x, styles) {
    const gpu = this.gpu;
    const w = this.w;
    const conv = (input, name, opts) => ops.conv2d(gpu, input, w[name], { name, ...opts });
    const step = (t, fn) => { const y = fn(t); t.release(); return y; };

    let h = conv(x, "enc.0", { k: 7, pad: 3, padMode: "reflect", act: "leaky" });
    h = step(h, (t) => conv(t, "enc.1", { k: 3, pad: 1, act: "leaky" }));
    h = step(h, (t) => conv(t, "enc.2", { k: 3, stride: 2, pad: 1, act: "leaky" }));
    const xs = step(h, (t) => conv(t, "enc.3", { k: 3, stride: 2, pad: 1, act: "leaky" }));

    for (let i = 0; i < 6; i++) {
      const p = `styles.${i}`;
      const a = conv(xs, p + ".conv1", { k: 3, pad: 1, padMode: "reflect" });
      const n1 = ops.adain(gpu, a, styles[2 * i]);
      let b;
      if (ops.needsRotation(w[p + ".conv2"])) {
        const r = ops.affine(gpu, a, n1, { mode: "relu", rotate: true });
        b = conv(r, p + ".conv2", { k: 3, pad: 1, padMode: "reflect", xr: r });
        r.release();
      } else {
        b = conv(a, p + ".conv2", { k: 3, pad: 1, padMode: "reflect", pro: n1, proRelu: true });
      }
      a.release();
      n1.release();
      const n2 = ops.adain(gpu, b, styles[2 * i + 1]);
      ops.affine(gpu, b, n2, { mode: "resid", out: xs });
      b.release();
      n2.release();
    }

    h = step(xs, (t) => conv(t, "dec.0", { k: 3, pad: 1, up: "bilinear", act: "leaky" }));
    h = step(h, (t) => conv(t, "dec.1", { k: 3, pad: 1, up: "bilinear", act: "leaky" }));
    h = step(h, (t) => conv(t, "dec.2", { k: 3, pad: 1, act: "leaky" }));
    return step(h, (t) => ops.convFew(gpu, t, w.out, "out"));
  }
}

function normalize(v) {
  let s = 0;
  for (const x of v) s += x * x;
  const inv = 1 / Math.sqrt(s);
  return Float32Array.from(v, (x) => x * inv);
}
