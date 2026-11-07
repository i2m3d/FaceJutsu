// What the WebNN face swappers (inswapper.js) share: a static graph for a batch of
// up to MAX_BATCH crops, with the aligned crops [B, S, S, 3] (NHWC, as the WebGPU side produces
// them) and the per-source data as inputs and the swapped crops [B, S, S, 3] in [0, 1] as output.
// The browser maps the graph onto the platform ML stack (on Windows: Windows ML / DirectML), which
// runs fp16 convolutions on the GPU's tensor cores, out of reach of the hand-written WebGPU
// kernels. Detection, alignment, recognition and paste-back stay on WebGPU; the crops cross over
// through the CPU. Opt-in (interop): the WebNN context shares the WebGPU device (Chrome's
// WebNN/WebGPU interop), the crops are written straight into the graph's input tensor and the
// swapped crops pasted straight from its output (forwardGPU, for batches that fit one graph).
// Correct, but in Chrome 155 every exportToGPU of a graph output stalls ~1.2 s (independent of the
// work queued), so it is off by default.
//
// Weights: the WebGPU swapper's loaded (direct, possibly quantized) weights are the source. When a
// graph is built, each one is decoded on the GPU (ConvRot rotation undone) into f16, read back and
// handed to the graph builder, which copies it: nothing is kept on the CPU, and only one graph
// (the current batch size) is kept, so weights exist once in WebNN's memory. (Device constant
// tensors would let graphs share them, but Chrome doesn't support them yet.)
//
// A subclass sets `size` (crop px), `sourceShape` and possibly `maxBatch`, and implements
// build(B, onProgress) (a graph with inputs "target" and "source" and output "out") and
// sourceData().

import { createWebNNContext, computeType, gpuInterop } from "./support.js";

// Largest batch per dispatch (a subclass may lower it: maxBatch); pixelshuffle 4 (16 crops) runs
// as 4 dispatches. A 16-crop inswapper graph was no faster than 4 x 4 crops with memory to spare
// (168 vs 158 ms), and ~5x slower once its activations no longer fit in VRAM.
export const MAX_BATCH = 4;

let half;
if (typeof Float16Array !== "undefined") {
  half = (f32) => new Uint16Array(new Float16Array(f32).buffer);
} else {
  const fb = new Float32Array(1), ub = new Uint32Array(fb.buffer);
  const bits = (v) => {
    fb[0] = v;
    const x = ub[0];
    const sign = (x >>> 16) & 0x8000;
    const e = ((x >>> 23) & 0xff) - 112;
    let m = x & 0x7fffff;
    if (e >= 31) return sign | 0x7c00 | (((x >>> 23) & 0xff) === 0xff && m ? 0x200 : 0);
    if (e <= 0) {
      if (e < -10) return sign;
      m |= 0x800000;
      const shift = 14 - e;
      let h = m >> shift;
      const rem = m & ((1 << shift) - 1), hf = 1 << (shift - 1);
      if (rem > hf || (rem === hf && h & 1)) h++;
      return sign | h;
    }
    let h = (e << 10) | (m >> 13);
    const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h++;
    return sign | h;
  };
  half = (f32) => Uint16Array.from(f32, bits);
}
export const toHalf = half;

export function fromHalf(u16) {
  if (typeof Float16Array !== "undefined") return Float32Array.from(new Float16Array(u16.buffer, u16.byteOffset, u16.length));
  return Float32Array.from(u16, (h) => {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  });
}

// typed data for a constant of type t from float values
export const floatData = (f32, t) => (t === "float16" ? toHalf(f32) : f32);

// MLOperand.dataType was a method in older Chromium builds and is an attribute now
export const typeOf = (o) => (typeof o.dataType === "function" ? o.dataType() : o.dataType);

export class WebNNSwapper {
  maxBatch = MAX_BATCH;

  // gpu + weights: the WebGPU swapper's GPU and its direct (non-Winograd) weights; st: the
  // SafeTensors file (biases, f32 tensors). precision: "float16" (default; float32 when
  // unsupported). interop: share the WebGPU device when the browser allows it (forwardGPU)
  static async load({ gpu, weights, st }, { precision = "float16", interop = false } = {}) {
    const m = new this();
    m.gpu = gpu;
    m.weights = weights;
    m.st = st;
    m.context = await createWebNNContext(interop ? gpu.device : null);
    m.interop = interop && gpuInterop(m.context);
    m.ct = computeType(m.context, precision);
    m.graph = null; // { B, graph, inputs, out, sourceOf }
    return m;
  }

  // The graph for batch size B (building it replaces the previous one).
  // onBuild(frac): progress while a graph is being built
  async ensureGraph(B, onBuild) {
    if (this.graph?.B === B) return this.graph;
    this.releaseGraph();
    const graph = await this.build(B, onBuild);
    const c = this.context;
    const crops = { dataType: "float32", shape: [B, this.size, this.size, 3] };
    const W = { writable: true };
    this.graph = {
      B,
      graph,
      inputs: {
        // with interop WebGPU writes the crops in and reads the results out directly
        target: this.interop ? await c.createExportableTensor(crops, this.gpu.device) : await c.createTensor({ ...crops, ...W }),
        source: await c.createTensor({ dataType: "float32", shape: this.sourceShape, ...W }),
      },
      out: this.interop ? await c.createExportableTensor(crops, this.gpu.device) : await c.createTensor({ ...crops, readable: true }),
      sourceOf: null,
    };
    return this.graph;
  }

  releaseGraph() {
    const g = this.graph;
    if (!g) return;
    for (const t of Object.values(g.inputs)) t.destroy();
    g.out.destroy();
    g.graph.destroy();
    this.graph = null;
  }

  writeSource(g, source) {
    if (g.sourceOf === source) return;
    this.context.writeTensor(g.inputs.source, source);
    g.sourceOf = source;
  }

  // crops: Float32Array [B, S, S, 3] in the swapper's input range; source: Float32Array from
  // sourceData() (the same object for the same source, so it is only written once) -> swapped
  // crops [B, S, S, 3] in [0, 1]. A last part short of the graph's batch is padded with zero
  // crops (pixelshuffle 3: 9 crops as 4 + 4 + 1 of 4, keeping the one graph of batch 4)
  async forward(crops, B, source, onBuild) {
    const b = Math.min(B, this.maxBatch);
    const g = await this.ensureGraph(b, onBuild);
    const c = this.context;
    this.writeSource(g, source);
    const per = crops.length / B;
    const out = B === b ? null : new Float32Array(crops.length);
    for (let i = 0; i < B; i += b) {
      const n = Math.min(b, B - i);
      let x = crops.subarray(i * per, (i + n) * per);
      if (n < b) {
        const padded = new Float32Array(b * per);
        padded.set(x);
        x = padded;
      }
      let y;
      if (this.interop) {
        // exportable tensors can't be written from the CPU: go through a WebGPU buffer
        const buf = await this.forwardGPU((input) => this.gpu.write(input, x), b, source);
        y = await this.gpu.read({ buf, size: x.length });
        buf.destroy();
      } else {
        c.writeTensor(g.inputs.target, x);
        c.dispatch(g.graph, g.inputs, { out: g.out });
        y = new Float32Array(await c.readTensor(g.out));
      }
      if (!out) return y;
      out.set(y.subarray(0, n * per), i * per);
    }
    return out;
  }

  // Interop path (this.interop), B <= maxBatch: fill(buf) records the WebGPU work that writes the
  // crops [B, S, S, 3] into the graph's input buffer; returns the output as a GPUBuffer of the
  // WebGPU device, lent by WebNN until the caller destroys it (after recording its reads).
  async forwardGPU(fill, B, source, onBuild) {
    if (B > this.maxBatch) throw new Error(`batch ${B} is over ${this.maxBatch}: use forward`);
    const g = await this.ensureGraph(B, onBuild);
    const c = this.context;
    this.writeSource(g, source);
    const input = await c.exportToGPU(g.inputs.target);
    fill(input);
    this.gpu.flush();
    input.destroy(); // back to WebNN, after the work recorded on it
    c.dispatch(g.graph, g.inputs, { out: g.out });
    return c.exportToGPU(g.out);
  }

  dispose() {
    this.releaseGraph();
    this.context.destroy?.();
  }
}
