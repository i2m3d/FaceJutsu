// Optional WebNN backend for the face detector (SCRFD-10G), used with the WebNN swapper: the
// detector's 58 small convolutions keep a WebGPU dispatch each busy for little work, which the
// platform ML stack (DirectML on Windows) runs fused and in fp16.
//
// Same network as models/scrfd.js, one static graph per input size: input the letterboxed
// [1, size, size, 3] tensor the WebGPU side produces (in `ct`: f16 halves the bytes that cross
// over), outputs the three [1, H, W, 30] head maps (NHWC, as models/scrfd.js returns them). Convs
// run in `ct` (fp16 when supported); the head outputs are cast back to fp32. The weights are small
// (4.2M parameters) and kept here.

import { LAYERS, convNames } from "../models/scrfd.js";
import { createWebNNContext, computeType } from "./support.js";

const toHalf = (f32) => new Uint16Array(new Float16Array(f32).buffer);
const floatData = (f32, t) => (t === "float16" ? toHalf(f32) : f32);
// MLOperand.dataType was a method in older Chromium builds and is an attribute now
const typeOf = (o) => (typeof o.dataType === "function" ? o.dataType() : o.dataType);
// 1x1: the FPN laterals and the stride-2 shortcuts; everything else 3x3
const kernelOf = (name) => (name.startsWith("lateral.") || name.endsWith(".down") ? 1 : 3);

export class WebNNSCRFD {
  // st: SafeTensors of a detector build. precision: "float16" (default; float32 when unsupported)
  static async load(st, { precision = "float16" } = {}) {
    if (typeof Float16Array === "undefined") throw new Error("this browser has no Float16Array");
    const m = new WebNNSCRFD();
    m.context = await createWebNNContext();
    m.ct = computeType(m.context, precision);
    m.convs = {};
    for (const name of convNames()) {
      const w = await st.f32(name + ".weight");
      const bias = await st.f32(name + ".bias");
      const k = kernelOf(name);
      const N = bias.length;
      m.convs[name] = {
        filter: { desc: { dataType: m.ct, shape: [N, k, k, w.length / (N * k * k)] }, data: floatData(w, m.ct) },
        bias: { desc: { dataType: m.ct, shape: [N] }, data: floatData(bias, m.ct) },
      };
    }
    m.graphs = new Map(); // input size -> { graph, input, outs }
    return m;
  }

  build(size) {
    const b = new MLGraphBuilder(this.context);
    const ct = this.ct;
    const cast = (x, t) => (typeOf(x) === t ? x : b.cast(x, t));
    const conv = (x, name, stride = 1) => {
      const c = this.convs[name];
      const p = kernelOf(name) === 3 ? 1 : 0;
      return b.conv2d(x, b.constant(c.filter.desc, c.filter.data), {
        padding: [p, p, p, p], strides: [stride, stride], bias: b.constant(c.bias.desc, c.bias.data), filterLayout: "ohwi",
      });
    };
    const relu = (x) => b.relu(x);
    const pool = { windowDimensions: [2, 2], strides: [2, 2] };
    const up = (x) => b.resample2d(x, { mode: "nearest-neighbor", scales: [2, 2], axes: [2, 3] });

    const input = b.input("input", { dataType: ct, shape: [1, size, size, 3] });
    let h = cast(b.transpose(input, { permutation: [0, 3, 1, 2] }), ct);
    h = relu(conv(h, "stem.0", 2));
    h = relu(conv(h, "stem.1"));
    h = relu(conv(h, "stem.2"));
    h = b.maxPool2d(h, pool);
    const feats = [];
    for (const [layer, blocks] of LAYERS) {
      for (let i = 0; i < blocks; i++) {
        const p = `layer${layer}.${i}`;
        const s = i === 0 && layer > 1 ? 2 : 1;
        const a = relu(conv(h, p + ".conv1", s));
        const sc = s === 2 ? conv(b.averagePool2d(h, pool), p + ".down") : h;
        h = relu(b.add(conv(a, p + ".conv2"), sc));
      }
      if (layer > 1) feats.push(h);
    }
    const [l0, l1, l2] = feats.map((f, i) => conv(f, `lateral.${i}`));
    const t1 = b.add(l1, up(l2));
    const t0 = b.add(l0, up(t1));
    const p3 = conv(t0, "fpn.0");
    let p4 = conv(t1, "fpn.1");
    let p5 = conv(l2, "fpn.2");
    p4 = b.add(p4, conv(p3, "down.0", 2));
    p5 = b.add(p5, conv(p4, "down.1", 2));
    p4 = conv(p4, "pafpn.0");
    p5 = conv(p5, "pafpn.1");
    const outs = {};
    [p3, p4, p5].forEach((f, s) => {
      for (let j = 0; j < 3; j++) f = relu(conv(f, `head.${s}.conv.${j}`));
      outs[`out${s}`] = b.transpose(cast(conv(f, `head.${s}.out`), "float32"), { permutation: [0, 2, 3, 1] });
    });
    return b.build(outs);
  }

  async ensureGraph(size) {
    let g = this.graphs.get(size);
    if (g) return g;
    const graph = await this.build(size);
    g = {
      graph,
      input: await this.context.createTensor({ dataType: this.ct, shape: [1, size, size, 3], writable: true }),
      outs: {},
    };
    for (let s = 0; s < 3; s++) {
      const n = size / (8 << s);
      g.outs[`out${s}`] = await this.context.createTensor({ dataType: "float32", shape: [1, n, n, 30], readable: true });
    }
    this.graphs.set(size, g);
    return g;
  }

  // x: [size * size * 3] letterboxed, normalized input in `ct` (Uint16Array of f16 bits, or
  // Float32Array) -> the three head maps
  async forward(x, size) {
    const g = await this.ensureGraph(size);
    this.context.writeTensor(g.input, x);
    this.context.dispatch(g.graph, { input: g.input }, g.outs);
    return Promise.all([0, 1, 2].map(async (s) => new Float32Array(await this.context.readTensor(g.outs[`out${s}`]))));
  }

  dispose() {
    for (const g of this.graphs.values()) {
      g.input.destroy();
      for (const t of Object.values(g.outs)) t.destroy();
      g.graph.destroy();
    }
    this.graphs.clear();
    this.context.destroy?.();
  }
}
