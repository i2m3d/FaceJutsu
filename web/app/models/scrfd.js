// SCRFD-10G face detector: letterboxed [1, 640, 640, 3] input ((RGB - 127.5) / 128)
// -> per stride 8 / 16 / 32 a [1, H, W, 30] head output: per anchor (2 per cell) a class logit,
// 4 box distances and 5 keypoint offsets, in units of the stride (see face.js decodeDetections).
//
//   backbone  3-conv stem, 2x2 max pool, ResNet-D basic blocks (3, 4, 2, 3); a stride-2 block's
//             shortcut is 2x2 avg pool + 1x1 conv
//   neck      PAFPN: 1x1 laterals, nearest top-down path, 3x3 fpn convs, stride-2 bottom-up path
//   heads     per stride three 3x3 conv + ReLU, then one 3x3 conv with the 30 stacked outputs

import * as ops from "../gpu/ops.js";

export const LAYERS = [[1, 3], [2, 4], [3, 2], [4, 3]];

export function convNames() {
  const names = ["stem.0", "stem.1", "stem.2"];
  for (const [layer, blocks] of LAYERS) {
    for (let b = 0; b < blocks; b++) {
      names.push(`layer${layer}.${b}.conv1`, `layer${layer}.${b}.conv2`);
      if (b === 0 && layer > 1) names.push(`layer${layer}.${b}.down`);
    }
  }
  names.push("lateral.0", "lateral.1", "lateral.2", "fpn.0", "fpn.1", "fpn.2", "down.0", "down.1", "pafpn.0", "pafpn.1");
  for (let s = 0; s < 3; s++) names.push(`head.${s}.conv.0`, `head.${s}.conv.1`, `head.${s}.conv.2`, `head.${s}.out`);
  return names;
}

export class SCRFD {
  static async load(gpu, st) {
    const m = new SCRFD(gpu);
    for (const n of convNames()) m.w[n] = await st.linear(gpu, n);
    return m;
  }

  constructor(gpu) {
    this.gpu = gpu;
    this.w = {};
  }

  forward(x) {
    const gpu = this.gpu;
    const conv = (input, name, opts) => ops.conv2d(gpu, input, this.w[name], { name: "det." + name, ...opts });
    const c3 = (input, name, opts = {}) => conv(input, name, { k: 3, pad: 1, ...opts });
    const step = (t, fn) => { const y = fn(t); t.release(); return y; };

    let h = c3(x, "stem.0", { stride: 2, act: "relu" });
    h = step(h, (t) => c3(t, "stem.1", { act: "relu" }));
    h = step(h, (t) => c3(t, "stem.2", { act: "relu" }));
    h = step(h, (t) => ops.pool2(gpu, t, "max"));
    const feats = [];
    for (const [layer, blocks] of LAYERS) {
      for (let b = 0; b < blocks; b++) {
        const p = `layer${layer}.${b}`;
        const s = b === 0 && layer > 1 ? 2 : 1;
        const a = c3(h, p + ".conv1", { stride: s, act: "relu" });
        let sc = h;
        if (s === 2) {
          const pooled = ops.pool2(gpu, h, "avg");
          sc = conv(pooled, p + ".down", { k: 1 });
          pooled.release();
        }
        const y = c3(a, p + ".conv2", { add: sc, act: "relu" });
        a.release();
        if (sc !== h) sc.release();
        if (!feats.includes(h)) h.release();
        h = y;
      }
      if (layer > 1) feats.push(h);
    }

    const [l0, l1, l2] = feats.map((f, i) => conv(f, `lateral.${i}`, { k: 1 }));
    for (const f of feats) f.release();
    ops.upAdd(gpu, l2, l1);
    ops.upAdd(gpu, l1, l0);
    const p3 = c3(l0, "fpn.0");
    let p4 = c3(l1, "fpn.1");
    let p5 = c3(l2, "fpn.2");
    for (const t of [l0, l1, l2]) t.release();
    p4 = step(p4, (t) => c3(p3, "down.0", { stride: 2, add: t }));
    p5 = step(p5, (t) => c3(p4, "down.1", { stride: 2, add: t }));
    p4 = step(p4, (t) => c3(t, "pafpn.0"));
    p5 = step(p5, (t) => c3(t, "pafpn.1"));

    return [p3, p4, p5].map((f, s) => {
      for (let j = 0; j < 3; j++) f = step(f, (t) => c3(t, `head.${s}.conv.${j}`, { act: "relu" }));
      return step(f, (t) => c3(t, `head.${s}.out`));
    });
  }
}
