// WebNN graph of inswapper_128 (~95% of the work per face), on the shared WebNN swapper
// (swapper.js): inputs the aligned crops [B, 128, 128, 3] in [0, 1] and the 12 AdaIN style rows
// [12, 2048]; output the swapped crops in the same layout.
//
// Numerics: convolutions run in `ct` (fp16 when supported). Instance norm + AdaIN and the residual
// stream stay fp32: AdaIN's eps of 1e-8 underflows in fp16, and the variance of large activations
// can overflow it. The graph runs NCHW, the layout DirectML prefers; filters stay in their stored
// OHWI layout (filterLayout "ohwi").
//
// Weights: the 3x3 filters are decoded from the WebGPU swapper's weights (swapper.js); the 7x7
// first and last convs are small and read from the file directly.

import * as ops from "../gpu/ops.js";
import { WebNNSwapper, floatData, fromHalf, typeOf } from "./swapper.js";

// [name, kernel size]
const CONVS = [
  ["enc.0", 7], ["enc.1", 3], ["enc.2", 3], ["enc.3", 3],
  ...[0, 1, 2, 3, 4, 5].flatMap((i) => [[`styles.${i}.conv1`, 3], [`styles.${i}.conv2`, 3]]),
  ["dec.0", 3], ["dec.1", 3], ["dec.2", 3], ["out", 7],
];

export class WebNNInswapper extends WebNNSwapper {
  size = 128;
  sourceShape = [12, 2048];

  // the graph's source input: the 12 style rows of styles() (WebGPU tensors), read back
  async sourceData(latent, styles) {
    const data = new Float32Array(12 * 2048);
    for (let i = 0; i < 12; i++) data.set(await this.gpu.read(styles[i]), i * 2048);
    return data;
  }

  // f16 bits (ct float16) or f32 values of a conv's dense OHWI filter
  async filterData(name, k) {
    if (k !== 3) return floatData(await this.st.f32(name + ".weight"), this.ct);
    const h = await ops.denseFiltersF16(this.gpu, this.weights[name]);
    return this.ct === "float16" ? h : fromHalf(h);
  }

  async build(B, onProgress) {
    const b = new MLGraphBuilder(this.context);
    const ct = this.ct;
    const scalars = new Map();
    const scalar = (v, t) => {
      const key = t + v;
      if (!scalars.has(key)) scalars.set(key, b.constant({ dataType: t, shape: [1] }, floatData(new Float32Array([v]), t)));
      return scalars.get(key);
    };
    const cast = (x, t) => (typeOf(x) === t ? x : b.cast(x, t));
    const filters = {};
    for (let i = 0; i < CONVS.length; i++) {
      const [name, k] = CONVS[i];
      const bias = await this.st.f32(name + ".bias");
      const N = bias.length;
      const data = await this.filterData(name, k);
      // constant() copies the data, so it isn't kept here
      filters[name] = {
        filter: b.constant({ dataType: ct, shape: [N, k, k, data.length / (N * k * k)] }, data),
        bias: b.constant({ dataType: ct, shape: [N] }, floatData(bias, ct)),
      };
      onProgress?.((i + 1) / (CONVS.length + 1));
    }
    const conv = (x, name, { stride = 1, pad = 0, reflect = false } = {}) => {
      const c = filters[name];
      let input = cast(x, ct);
      if (reflect) input = b.pad(input, [0, 0, pad, pad], [0, 0, pad, pad], { mode: "reflection" });
      const p = reflect ? 0 : pad;
      return b.conv2d(input, c.filter, { padding: [p, p, p, p], strides: [stride, stride], bias: c.bias, filterLayout: "ohwi" });
    };
    const lrelu = (x) => b.leakyRelu(x, { alpha: 0.2 });
    const styles = b.input("source", { dataType: "float32", shape: [12, 2048] });
    const row = (i, off) => b.reshape(b.slice(styles, [i, off], [1, 1024]), [1024]);
    // instance norm (eps 1e-8) with the AdaIN scale/shift of style row i, in fp32
    const adain = (x, i) => b.instanceNormalization(cast(x, "float32"), { scale: row(i, 0), bias: row(i, 1024), epsilon: 1e-8 });

    const target = b.input("target", { dataType: "float32", shape: [B, 128, 128, 3] });
    let h = b.transpose(target, { permutation: [0, 3, 1, 2] });
    h = lrelu(conv(h, "enc.0", { pad: 3, reflect: true }));
    h = lrelu(conv(h, "enc.1", { pad: 1 }));
    h = lrelu(conv(h, "enc.2", { stride: 2, pad: 1 }));
    let x = cast(lrelu(conv(h, "enc.3", { stride: 2, pad: 1 })), "float32");
    for (let i = 0; i < 6; i++) {
      let a = b.relu(adain(conv(x, `styles.${i}.conv1`, { pad: 1, reflect: true }), 2 * i));
      a = adain(conv(a, `styles.${i}.conv2`, { pad: 1, reflect: true }), 2 * i + 1);
      x = b.add(x, a);
    }
    h = b.resample2d(cast(x, ct), { mode: "linear", scales: [2, 2], axes: [2, 3] });
    h = lrelu(conv(h, "dec.0", { pad: 1 }));
    h = b.resample2d(h, { mode: "linear", scales: [2, 2], axes: [2, 3] });
    h = lrelu(conv(h, "dec.1", { pad: 1 }));
    h = lrelu(conv(h, "dec.2", { pad: 1 }));
    h = cast(conv(h, "out", { pad: 3, reflect: true }), "float32");
    h = b.add(b.mul(b.tanh(h), scalar(0.5, "float32")), scalar(0.5, "float32"));
    const graph = await b.build({ out: b.transpose(h, { permutation: [0, 2, 3, 1] }) });
    onProgress?.(1);
    return graph;
  }
}
