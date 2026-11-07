// Reads safetensors files (a whole file, as an ArrayBuffer) and uploads weights.
//
// Matmul weight objects: { kind: "f32" | "bf16" | "f16" | "i8" | "w4", N, K, buf, scale?, srel?, codebook?, bias? }
// Convolution weights are the same objects with K = KH*KW*Cin. Quantized
// kinds follow the ComfyUI comfy_quant layout (comfy_kitchen int8_tensorwise / asym_w4a8_int8
// with ConvRot).

export function bf16ToF32(u8) {
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, u8.byteLength / 2);
  const out = new Float32Array(u16.length);
  const o32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) o32[i] = u16[i] << 16;
  return out;
}

export class SafeTensors {
  constructor(file) {
    const n = Number(new DataView(file).getBigUint64(0, true));
    this.file = file;
    this.header = JSON.parse(new TextDecoder().decode(new Uint8Array(file, 8, n)));
    delete this.header.__metadata__;
    this.dataStart = 8 + n;
  }

  has(name) {
    return name in this.header;
  }

  info(name) {
    const t = this.header[name];
    if (!t) throw new Error(`missing tensor ${name}`);
    return t;
  }

  // A copy of its own: tensors aren't aligned in the file, and f32() needs them to be.
  async bytes(name) {
    const t = this.info(name);
    const [a, b] = t.data_offsets;
    return new Uint8Array(this.file.slice(this.dataStart + a, this.dataStart + b));
  }

  async f32(name) {
    const t = this.info(name);
    const u8 = await this.bytes(name);
    if (t.dtype === "F32") return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
    if (t.dtype === "BF16") return bf16ToF32(u8);
    throw new Error(`unsupported dtype ${t.dtype} for ${name}`);
  }

  async vector(gpu, name) {
    return gpu.upload(await this.f32(name));
  }

  // "<base>.weight" (+ comfy_quant companions) and "<base>.bias" when present.
  async linear(gpu, base) {
    const w = this.info(base + ".weight");
    const bias = this.has(base + ".bias") ? await this.vector(gpu, base + ".bias") : null;
    const N = w.shape[0];
    const name = base;
    if (this.has(base + ".comfy_quant")) {
      const meta = JSON.parse(new TextDecoder().decode(await this.bytes(base + ".comfy_quant")));
      if (!meta.convrot) throw new Error(`${base}: only ConvRot-quantized weights are supported`);
      if (meta.format === "int8_tensorwise") {
        return { name, kind: "i8", N, K: w.shape[1], buf: gpu.upload(await this.bytes(base + ".weight")), scale: await this.vector(gpu, base + ".weight_scale"), bias };
      }
      if (meta.format === "asym_w4a8_int8") {
        return {
          name, kind: "w4", N, K: w.shape[1] * 2,
          buf: gpu.upload(await this.bytes(base + ".weight")),
          srel: gpu.upload(await this.bytes(base + ".weight_s_rel")),
          scale: await this.vector(gpu, base + ".weight_s_channel"),
          codebook: await this.vector(gpu, base + ".weight_codebook"),
          bias,
        };
      }
      throw new Error(`unsupported quant format ${meta.format}`);
    }
    const K = w.shape.slice(1).reduce((a, b) => a * b, 1);
    if (w.dtype === "F32") return { name, kind: "f32", N, K, buf: gpu.upload(await this.bytes(base + ".weight")), bias };
    if (w.dtype === "BF16") return { name, kind: "bf16", N, K, buf: gpu.upload(await this.bytes(base + ".weight")), bias };
    if (w.dtype === "F16") return { name, kind: "f16", N, K, buf: gpu.upload(await this.bytes(base + ".weight")), bias };
    throw new Error(`unsupported weight dtype ${w.dtype} for ${base}`);
  }
}

const SNAKE = [0, 1, 3, 2]; // stored position of 2x2 tap ty*2 + tx

// The stored 3x3 kernel of a stride-2 transposed conv [Cout, 9*Cin] (tap ky*3 + kx; output pixel
// 2i + k from input pixel i) -> a 2x2 conv (pad 1) to 4*Cout channels [4*Cout, 4*Cin],
// polyphase: transposed-conv output pixel (2a + py, 2b + px) is conv pixel (a, b), channels
// (py*2 + px)*Cout.. . Output row 2a + py reads input row a - 1 + ty with kernel row
// py + 2 - 2*ty; past the kernel the tap is zero (7 of 16). Taps are stored in snake order, (0,0),
// (0,1), (1,1), (1,0), so that each phase's non-zero taps are one k range for the GEMM to run
// (gemm.js phases). It only moves stored elements, so it works for every stored format but 4-bit:
// int8 rows keep their scale, and ConvRot rotates each tap's input channels on their own. A
// stored bias [Cout] is repeated for the 4 phases.
export async function polyphase(gpu, st, name) {
  const info = st.info(name + ".weight");
  const [N, K] = info.shape;
  const cin = K / 9;
  const es = { F32: 4, F16: 2, BF16: 2, I8: 1 }[info.dtype];
  if (!es) throw new Error(`${name}: can't expand a ${info.dtype} weight`);
  const src = await st.bytes(name + ".weight");
  const tap = cin * es;
  const dst = new Uint8Array(16 * N * tap);
  for (let q = 0; q < 4; q++) {
    for (let t = 0; t < 4; t++) {
      const ky = (q >> 1) + 2 - 2 * (t >> 1);
      const kx = (q & 1) + 2 - 2 * (t & 1);
      if (ky > 2 || kx > 2) continue;
      for (let co = 0; co < N; co++) {
        const from = (co * 9 + ky * 3 + kx) * tap;
        dst.set(src.subarray(from, from + tap), ((q * N + co) * 4 + SNAKE[t]) * tap);
      }
    }
  }
  const four = (v) => {
    const out = new Float32Array(4 * N);
    for (let q = 0; q < 4; q++) out.set(v, q * N);
    return gpu.upload(out);
  };
  const bias = st.has(name + ".bias") ? four(await st.f32(name + ".bias")) : null;
  const W = { name, N: 4 * N, K: 4 * cin, buf: gpu.upload(dst), bias };
  if (info.dtype !== "I8") return { ...W, kind: info.dtype.toLowerCase() };
  return { ...W, kind: "i8", scale: four(await st.f32(name + ".weight_scale")) };
}

// Frees every GPU buffer reachable from a loaded model object.
export function destroyWeights(o, seen = new Set()) {
  if (!o || typeof o !== "object" || seen.has(o)) return;
  seen.add(o);
  if (o instanceof GPUBuffer) { o.destroy(); return; }
  if (o instanceof SafeTensors || ArrayBuffer.isView(o)) return;
  for (const v of Array.isArray(o) ? o : Object.values(o)) destroyWeights(v, seen);
}
