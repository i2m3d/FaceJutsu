// ArcFace IResNet-50 (trained on WebFace600K): aligned [B, 112, 112, 3] RGB crops in
// [-1, 1] -> [B, 512] identity embeddings (not normalized).
//
//   stem    3x3 conv + PReLU
//   blocks  (3, 4, 14, 3) IBasicBlocks: BN -> 3x3 conv -> PReLU -> 3x3 conv (stride 2 in a layer's
//           first block) + shortcut (1x1 stride-2 conv there, identity elsewhere). The BN after
//           each conv is folded into it; the BN in front of conv1 is a per-channel scale/shift
//           applied while conv1 loads its input (zero padding has to stay zero).
//   head    flatten (NHWC) -> fc 25088 -> 512, both BatchNorms folded in

import * as ops from "../gpu/ops.js";

const LAYERS = [3, 4, 14, 3];

export class ArcFace {
  static async load(gpu, st, onProgress) {
    const m = new ArcFace(gpu);
    const w = m.w;
    w.conv0 = await st.linear(gpu, "conv0");
    w.prelu0 = await st.vector(gpu, "prelu0");
    for (let l = 0; l < LAYERS.length; l++) {
      for (let b = 0; b < LAYERS[l]; b++) {
        const p = `layers.${l}.${b}`;
        const scale = await st.f32(p + ".bn1.scale");
        const bn = new Float32Array(scale.length * 2);
        bn.set(scale);
        bn.set(await st.f32(p + ".bn1.shift"), scale.length);
        w[p] = {
          bn: gpu.upload(bn), C: scale.length,
          conv1: await st.linear(gpu, p + ".conv1"),
          prelu: await st.vector(gpu, p + ".prelu"),
          conv2: await st.linear(gpu, p + ".conv2"),
          down: b === 0 ? await st.linear(gpu, p + ".down") : null,
        };
      }
      onProgress?.((l + 1) / (LAYERS.length + 1));
    }
    w.fc = await st.linear(gpu, "fc");
    onProgress?.(1);
    return m;
  }

  constructor(gpu) {
    this.gpu = gpu;
    this.w = {};
  }

  forward(x) {
    const gpu = this.gpu;
    const w = this.w;
    let h = ops.conv2d(gpu, x, w.conv0, { k: 3, pad: 1, act: "prelu", prelu: w.prelu0, name: "arc.conv" });
    for (let l = 0; l < LAYERS.length; l++) {
      for (let b = 0; b < LAYERS[l]; b++) {
        const blk = w[`layers.${l}.${b}`];
        const bn = { buf: blk.bn, off: blk.C };
        let a;
        if (ops.needsRotation(blk.conv1)) {
          const r = ops.affine(gpu, h, bn, { mode: "plain", rotate: true, proStride: 0 });
          a = ops.conv2d(gpu, r, blk.conv1, { k: 3, pad: 1, act: "prelu", prelu: blk.prelu, xr: r, name: "arc.conv" });
          r.release();
        } else {
          a = ops.conv2d(gpu, h, blk.conv1, { k: 3, pad: 1, act: "prelu", prelu: blk.prelu, pro: bn, proStride: 0, name: "arc.conv" });
        }
        const sc = blk.down ? ops.conv2d(gpu, h, blk.down, { k: 1, stride: 2, name: "arc.down" }) : h;
        const y = ops.conv2d(gpu, a, blk.conv2, { k: 3, pad: 1, stride: blk.down ? 2 : 1, add: sc, name: "arc.conv" });
        a.release();
        if (sc !== h) sc.release();
        h.release();
        h = y;
      }
    }
    const B = h.shape[0];
    h.shape = [B, h.size / B];
    const e = ops.linear(gpu, h, w.fc, { name: "arc.fc" });
    h.release();
    return e;
  }
}
