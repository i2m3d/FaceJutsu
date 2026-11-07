// the community-trained XSeg (SN256_XSeg, the occlusion mask): aligned RGB face crops [B, 256, 256, 3] in
// [0, 1] -> what of each crop is face, [B, 256, 256, 1] in [0, 1] (hands, glasses, hair in front
// of the face: 0).
//
//   encoder     6 levels of 3x3 convs (32 .. 256 channels), each level then blurred and
//               downsampled 2x by a fixed filter ([1,3,3,1] and [1,2,1] per axis, then 2x2 boxes)
//   bottleneck  dense 4096 -> 512 -> 4096 on the 4 px map
//   decoder     per level the stride-2 transposed conv, run as a polyphase 2x2 conv to 4x the
//               channels (as the enhancer's), concatenated with the encoder's map of that size, then 3x3
//               convs; a 3x3 conv to the mask, sigmoid
//
// Every conv and transposed conv is followed by FRN + TLU: per image and channel, the mean of x^2
// (one reduction), then one kernel that applies max(x*S + T, tau) and writes the result the way
// its consumer reads it: ConvRot-rotated for a quantized conv, and for a transposed conv
// pixel-shuffled with the encoder's map appended.

import * as ops from "../gpu/ops.js";
import { polyphase } from "../weights.js";

export const ENCODER = [["conv01", "conv02"], ["conv11", "conv12"], ["conv21", "conv22"], ["conv31", "conv32", "conv33"],
  ["conv41", "conv42", "conv43"], ["conv51", "conv52", "conv53"]];
export const DECODER = [["up5", ["uconv53", "uconv52", "uconv51"]], ["up4", ["uconv43", "uconv42", "uconv41"]], ["up3", ["uconv33", "uconv32", "uconv31"]],
  ["up2", ["uconv22", "uconv21"]], ["up1", ["uconv12", "uconv11"]], ["up0", ["uconv02", "uconv01"]]];
// the downsampling blur of encoder levels 0 and 1 (1D taps); the deeper ones are 2x2 boxes
export const BLURS = [[1, 3, 3, 1], [1, 2, 1]];

// px of each 3x3 conv's map
const RES = Object.fromEntries([...ENCODER.map((names, i) => names.map((n) => [n, 256 >> i])), ...DECODER.map(([, names], i) => names.map((n) => [n, 8 << i]))].flat());
// The 3x3 convs at these sizes run as Winograd F(2x2, 3x3) (inswapper.js), but for the 384-channel
// ones after the concats (the filter preparation takes up to 256): -3 ms per face. At 128 and
// 256 px (32 and 64 channels) the direct conv is faster.
const WINOGRAD = [8, 64]; // px

export class XSeg {
  static async load(gpu, st, onProgress, { winograd = true } = {}) {
    const m = new XSeg(gpu);
    const ups = new Set(DECODER.map(([up]) => up));
    const names = [...ENCODER.flat(), ...DECODER.flatMap(([up, convs]) => [up, ...convs])];
    for (let i = 0; i < names.length; i++) {
      const n = names[i];
      m.w[n] = ups.has(n) ? await polyphase(gpu, st, n) : await st.linear(gpu, n);
      if (winograd && RES[n] >= WINOGRAD[0] && RES[n] <= WINOGRAD[1] && m.w[n].K <= 9 * 256) {
        // once the GPU runs out of memory for Winograd filters, the remaining convs stay direct
        const wino = await ops.winogradFilters(gpu, m.w[n]);
        if (wino) m.w[n] = wino;
        else {
          console.warn(`not enough GPU memory for Winograd filters from ${n} on: direct convolutions`);
          winograd = false;
        }
      }
      m.w[n].frn = { buf: await st.vector(gpu, n + ".frn"), eps: (await st.f32(n + ".eps"))[0] };
      onProgress?.((i + 1) / (names.length + 2));
    }
    for (const n of ["dense1", "dense2"]) m.w[n] = await st.linear(gpu, n);
    m.w.out = { k: 3, N: 1, wt: gpu.upload(Float32Array.from(await st.f32("out.weight"))), bias: await st.vector(gpu, "out.bias") };
    onProgress?.(1);
    return m;
  }

  constructor(gpu) {
    this.gpu = gpu;
    this.w = {};
    this.crop = { size: 256, mul: 1 / 255, add: 0 }; // input crops in [0, 1]
  }

  // x: [B, 256, 256, 3] in [0, 1] -> XSeg's output [B, 256, 256, 1]. taps: an object that receives
  // copies of the encoder's maps (enc.0 .. enc.5), the bottleneck and the decoder's maps (dec.5 ..
  // dec.0).
  forward(x, taps = null) {
    const gpu = this.gpu;
    const w = this.w;
    const B = x.shape[0];
    const tap = (name, t, rotated) => {
      if (!taps) return;
      if (rotated) taps[name] = ops.rotate(gpu, t); // a rotation is its own inverse
      else {
        taps[name] = gpu.empty(t.shape);
        gpu.encode((enc) => enc.copyBufferToBuffer(t.buf, 0, taps[name].buf, 0, t.size * 4));
      }
    };
    // a conv's output through its FRN + TLU, rotated when its consumer (the weight `next`) is
    // quantized; h is rotated whenever its reader needs it
    let rot = false;
    const frn = (y, name, next, opts = {}) => {
      rot = !!next && ops.needsRotation(next);
      const pro = ops.frn(gpu, y, w[name].frn.buf, w[name].frn.eps, opts.shuffle ? 4 : 1);
      const h = ops.frnOut(gpu, y, pro, { ...opts, rotate: rot });
      pro.release();
      y.release();
      return h;
    };
    const conv = (h, name) => ops.conv2d(gpu, h, w[name], { k: 3, pad: 1, xr: rot ? h : null, name: `xseg.conv${h.shape[1]}` });

    let h = x;
    const skips = [];
    ENCODER.forEach((names, i) => {
      names.forEach((n, j) => {
        const y = conv(h, n);
        if (h !== x) h.release();
        h = frn(y, n, w[names[j + 1]]);
      });
      skips.push(h);
      tap(`enc.${i}`, h, false);
      h = i < BLURS.length ? ops.blurDown(gpu, h, BLURS[i]) : ops.pool2(gpu, h, "avg");
    });

    // bottleneck: dense1 on the 4 px map's NCHW flatten, as exported; dense2's rows are stored
    // in NHWC order
    const f = ops.flattenNCHW(gpu, h);
    h.release();
    const z = ops.linear(gpu, f, w.dense1, { name: "xseg.dense" });
    f.release();
    h = ops.linear(gpu, z, w.dense2, { name: "xseg.dense" });
    z.release();
    h.shape = [B, 4, 4, h.shape[1] / 16];
    tap("bottleneck", h, false);

    DECODER.forEach(([up, names], i) => {
      const u = ops.conv2d(gpu, h, w[up], { k: 2, pad: 1, padHi: 0, phases: w[up].N / 4, xr: rot ? h : null, name: `xseg.up${2 * h.shape[1]}` });
      h.release();
      const skip = skips.pop();
      h = frn(u, up, w[names[0]], { shuffle: true, skip });
      skip.release();
      names.forEach((n, j) => {
        const y = conv(h, n);
        h.release();
        h = frn(y, n, j + 1 < names.length ? w[names[j + 1]] : DECODER[i + 1]?.[0] && w[DECODER[i + 1][0]]);
      });
      tap(`dec.${up.slice(2)}`, h, rot);
    });
    const y = ops.convFew(gpu, h, w.out, "xseg.out", { pad: "zeros", act: "sigmoid" });
    h.release();
    return y;
  }
}
