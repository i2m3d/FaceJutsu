// Optional WebNN backend for the occlusion mask (the community-trained XSeg, models/xseg.js), used with
// the WebNN swapper: XSeg's 3x3 convs at 128 and 256 px have only 32 and 64 output channels, which
// fill half of the WebGPU GEMM's tiles; the platform ML stack (DirectML on Windows) runs them on
// the tensor cores in fp16.
//
// Same network as models/xseg.js, one static graph for one 256 px crop: input the crop
// [1, 256, 256, 3] in [0, 1] (NHWC, as the WebGPU side produces it; in `ct`, f16 halves the bytes
// that cross over), output XSeg's sigmoid [1, 256, 256, 1] in fp32. The graph runs NCHW. Convs,
// the transposed convs and the dense layers run in `ct` (fp16 when supported); each FRN's
// statistics in fp32 (the mean of x^2 can overflow fp16), its per-channel scale and the TLU in
// `ct` (all fp32: 1.3x the time). The transposed convs run as WebNN's convTranspose2d on the
// stored 3x3 kernels, cropped to exactly 2x by padding only after.
//
// Weights: the quantized ones are decoded on the GPU (ConvRot rotation undone) into dense f16 and
// read back (ops.denseWeightsF16); the graph builder copies them, so nothing is kept here.

import * as ops from "../gpu/ops.js";
import { ENCODER, DECODER, BLURS } from "../models/xseg.js";
import { createWebNNContext, computeType } from "./support.js";
import { floatData, fromHalf, typeOf } from "./swapper.js";
import { destroyWeights } from "../weights.js";

const N = 256; // crop px

export class WebNNXSeg {
  // gpu: the WebGPU device (decodes the quantized weights); st: SafeTensors of an XSeg build.
  // precision: "float16" (default; float32 when unsupported)
  static async load(gpu, st, { precision = "float16" } = {}) {
    if (typeof Float16Array === "undefined") throw new Error("this browser has no Float16Array");
    const m = new WebNNXSeg();
    m.gpu = gpu;
    m.st = st;
    m.context = await createWebNNContext();
    m.ct = computeType(m.context, precision);
    m.crop = { size: N, mul: 1 / 255, add: 0 }; // as models/xseg.js
    m.graph = null;
    return m;
  }

  // f16 bits (ct float16) or f32 values of a stored weight [N, K] (convs: OHWI)
  async weightData(name) {
    const info = this.st.info(name + ".weight");
    if (info.dtype === "F16" && !this.st.has(name + ".comfy_quant")) {
      const u8 = await this.st.bytes(name + ".weight");
      const h = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
      return this.ct === "float16" ? h : fromHalf(h);
    }
    if (info.dtype === "F32" || info.dtype === "BF16") return floatData(await this.st.f32(name + ".weight"), this.ct);
    // quantized: through the WebGPU decoder
    const W = await this.st.linear(this.gpu, name);
    const h = await ops.denseWeightsF16(this.gpu, W);
    destroyWeights(W);
    return this.ct === "float16" ? h : fromHalf(h);
  }

  async build(onProgress) {
    const b = new MLGraphBuilder(this.context);
    const ct = this.ct;
    const st = this.st;
    const cast = (x, t) => (typeOf(x) === t ? x : b.cast(x, t));
    const f32 = (shape, data) => b.constant({ dataType: "float32", shape }, data);
    const names = [...ENCODER.flat(), ...DECODER.flatMap(([up, convs]) => [up, ...convs]), "dense1", "dense2", "out"];
    const W = {};
    for (let i = 0; i < names.length; i++) {
      const n = names[i];
      const bias = await st.f32(n + ".bias");
      const data = n === "out" ? floatData(await st.f32("out.weight"), ct) : await this.weightData(n);
      const Cout = bias.length;
      const shape = n.startsWith("dense") ? [Cout, data.length / Cout] : [Cout, 3, 3, data.length / (9 * Cout)];
      W[n] = { w: b.constant({ dataType: ct, shape }, data), bias: b.constant({ dataType: ct, shape: [Cout] }, floatData(bias, ct)) };
      if (st.has(n + ".frn")) {
        const p = await st.f32(n + ".frn");
        const row = (j, t) => b.constant({ dataType: t, shape: [1, Cout, 1, 1] }, floatData(p.slice(j * Cout, (j + 1) * Cout), t));
        W[n].frn = { gamma: row(0, "float32"), beta: row(1, ct), tau: row(2, ct), eps: f32([1], Float32Array.from(await st.f32(n + ".eps"))) };
      }
      onProgress?.((i + 1) / (names.length + 1));
    }
    // conv (3x3, pad 1) or transposed conv (3x3, stride 2, cropped to 2x), then FRN + TLU in fp32,
    // back in ct
    // FRN + TLU: the statistics in fp32, the per-channel scale applied in ct
    const frn = (y, n) => {
      const { gamma, beta, tau, eps } = W[n].frn;
      const x = cast(y, "float32");
      const nu2 = b.reduceMean(b.mul(x, x), { axes: [2, 3], keepDimensions: true });
      const S = cast(b.mul(b.reciprocal(b.sqrt(b.add(nu2, eps))), gamma), ct);
      return b.max(b.add(b.mul(y, S), beta), tau);
    };
    const conv = (h, n) => frn(b.conv2d(h, W[n].w, { padding: [1, 1, 1, 1], bias: W[n].bias, filterLayout: "ohwi" }), n);
    const up = (h, n) => frn(b.convTranspose2d(h, W[n].w, { strides: [2, 2], padding: [0, 1, 0, 1], bias: W[n].bias, filterLayout: "ohwi" }), n);
    // the encoder's blurred stride-2 downsampling: depthwise [1,3,3,1] / [1,2,1] filters (zero pad
    // before / after as the reference's), then 2x2 boxes
    const down = (h, i, C) => {
      if (i >= BLURS.length) return b.averagePool2d(h, { windowDimensions: [2, 2], strides: [2, 2] });
      const k = BLURS[i];
      const s = k.reduce((a, v) => a + v, 0);
      const f = new Float32Array(C * k.length * k.length);
      for (let c = 0; c < C; c++) for (let y = 0; y < k.length; y++) for (let x = 0; x < k.length; x++) f[(c * k.length + y) * k.length + x] = (k[y] * k[x]) / (s * s);
      const lo = (k.length - 1) >> 1;
      const hi = k.length - 1 - lo;
      return b.conv2d(h, b.constant({ dataType: ct, shape: [C, 1, k.length, k.length] }, floatData(f, ct)),
        { padding: [lo, hi, lo, hi], strides: [2, 2], groups: C, filterLayout: "oihw" });
    };

    const input = b.input("input", { dataType: ct, shape: [1, N, N, 3] });
    let h = b.transpose(input, { permutation: [0, 3, 1, 2] });
    const skips = [];
    ENCODER.forEach((convs, i) => {
      for (const n of convs) h = conv(h, n);
      skips.push(h);
      h = down(h, i, st.info(convs.at(-1) + ".bias").shape[0]);
    });
    // bottleneck: dense1 on the NCHW flatten (the graph's layout); dense2's rows are NHWC
    let z = b.gemm(b.reshape(h, [1, 4096]), W.dense1.w, { bTranspose: true, c: W.dense1.bias });
    z = b.gemm(z, W.dense2.w, { bTranspose: true, c: W.dense2.bias });
    h = b.transpose(b.reshape(z, [1, 4, 4, 256]), { permutation: [0, 3, 1, 2] });
    for (const [u, convs] of DECODER) {
      h = b.concat([up(h, u), skips.pop()], 1);
      for (const n of convs) h = conv(h, n);
    }
    const y = b.sigmoid(cast(b.conv2d(h, W.out.w, { padding: [1, 1, 1, 1], bias: W.out.bias, filterLayout: "ohwi" }), "float32"));
    const graph = await b.build({ out: b.reshape(y, [1, N, N, 1]) });
    onProgress?.(1);
    return graph;
  }

  async ensureGraph(onBuild) {
    if (this.graph) return this.graph;
    const graph = await this.build(onBuild);
    this.graph = {
      graph,
      input: await this.context.createTensor({ dataType: this.ct, shape: [1, N, N, 3], writable: true }),
      out: await this.context.createTensor({ dataType: "float32", shape: [1, N, N, 1], readable: true }),
    };
    return this.graph;
  }

  // x: [B, 256, 256, 3] crops in [0, 1] (a WebGPU tensor) -> XSeg's output [B, 256, 256, 1] (a
  // WebGPU tensor), one crop per dispatch
  async forward(x) {
    const g = await this.ensureGraph();
    const B = x.shape[0];
    const crops = this.ct === "float16" ? await ops.readHalf(this.gpu, x) : await this.gpu.read(x);
    const per = N * N * 3;
    const out = new Float32Array(B * N * N);
    for (let i = 0; i < B; i++) {
      this.context.writeTensor(g.input, crops.subarray(i * per, (i + 1) * per));
      this.context.dispatch(g.graph, { input: g.input }, { out: g.out });
      out.set(new Float32Array(await this.context.readTensor(g.out)), i * N * N);
    }
    return this.gpu.fromArray(out, [B, N, N, 1]);
  }

  dispose() {
    if (this.graph) {
      this.graph.input.destroy();
      this.graph.out.destroy();
      this.graph.graph.destroy();
      this.graph = null;
    }
    this.context.destroy?.();
  }
}
