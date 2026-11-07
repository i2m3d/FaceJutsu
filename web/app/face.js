// Face geometry on the CPU: alignment templates, similarity fits, detector output decoding and
// the paste-back mask.
//
// Affine maps are 2x3 row-major arrays [a, b, c, d, e, f]: (x, y) -> (a*x + b*y + c, d*x + e*y + f),
// with pixel centers at integer coordinates (as cv2.warpAffine).

// 5-point templates (left eye, right eye, nose, left mouth corner, right mouth corner) as
// fractions of the crop size
export const RECOGNIZE_112 = [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.500225, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]];
export const SWAP_128 = [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]];
// the wider framing for the face enhancers that see the whole head (Enhancer-512)
export const WIDE_512 = [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904], [0.39308822, 0.725411], [0.61150205, 0.72490465]];
export const TEMPLATES = { recognize_112: RECOGNIZE_112, swap_128: SWAP_128, wide_512: WIDE_512 };

// Detector input size (square). Any multiple of 32 works; smaller is faster (320: a quarter of
// the work) but misses small faces.
export const DET_SIZE = 640;

// Least-squares similarity transform (rotation, uniform scale, translation) mapping the src
// points onto dst.
export function similarity(src, dst) {
  const n = src.length;
  let msx = 0, msy = 0, mdx = 0, mdy = 0;
  for (let i = 0; i < n; i++) { msx += src[i][0]; msy += src[i][1]; mdx += dst[i][0]; mdy += dst[i][1]; }
  msx /= n; msy /= n; mdx /= n; mdy /= n;
  let ss = 0, sa = 0, sb = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - msx, sy = src[i][1] - msy, dx = dst[i][0] - mdx, dy = dst[i][1] - mdy;
    ss += sx * sx + sy * sy;
    sa += sx * dx + sy * dy;
    sb += sx * dy - sy * dx;
  }
  const a = sa / ss, b = sb / ss;
  return [a, -b, mdx - (a * msx - b * msy), b, a, mdy - (b * msx + a * msy)];
}

export function invert(m) {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d;
  return [e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det];
}

export function apply(m, x, y) {
  return [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
}

// Maps an image's 5 landmarks onto a template scaled to `size`: -> image->crop matrix.
export function alignment(kps, template, size) {
  return similarity(kps, template.map(([x, y]) => [x * size, y * size]));
}

// Scale of a similarity map (source pixels per destination pixel for an output->source map)
export const scaleOf = (m) => Math.hypot(m[0], m[3]);

// Detector letterbox: the image is scaled to fit size x size at the top left (both up and down,
//). Returns the output->image resampling map and the valid region.
export function letterbox(w, h, size = DET_SIZE) {
  const s = Math.min(size / w, size / h);
  return {
    scale: s,
    vw: Math.min(size, Math.round(w * s)),
    vh: Math.min(size, Math.round(h * s)),
    m: [1 / s, 0, 0.5 / s - 0.5, 0, 1 / s, 0.5 / s - 0.5],
  };
}

// Head outputs (Float32Array [H*W*30] per stride 8/16/32, rows in (y, x) order) -> faces
// [{ box: [x1, y1, x2, y2], score, kps: [[x, y] x 5] }] in image pixels, after NMS.
export function decodeDetections(outs, scale, { threshold = 0.5, nms = 0.4, size = DET_SIZE } = {}) {
  const found = [];
  [8, 16, 32].forEach((stride, i) => {
    const o = outs[i];
    const n = size / stride;
    for (let cell = 0; cell < n * n; cell++) {
      const base = cell * 30;
      const cx = (cell % n) * stride;
      const cy = Math.floor(cell / n) * stride;
      for (let a = 0; a < 2; a++) {
        const score = 1 / (1 + Math.exp(-o[base + a]));
        if (score < threshold) continue;
        const r = base + 2 + 4 * a;
        const k = base + 10 + 10 * a;
        const box = [cx - o[r] * stride, cy - o[r + 1] * stride, cx + o[r + 2] * stride, cy + o[r + 3] * stride].map((v) => v / scale);
        const kps = [];
        for (let p = 0; p < 5; p++) kps.push([(cx + o[k + 2 * p] * stride) / scale, (cy + o[k + 2 * p + 1] * stride) / scale]);
        found.push({ box, score, kps });
      }
    }
  });
  found.sort((p, q) => q.score - p.score);
  const keep = [];
  for (const f of found) if (keep.every((k) => iou(f.box, k.box) <= nms)) keep.push(f);
  return keep;
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / Math.max(union, 1e-9);
}

export const area = (f) => (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);

// The kernel of cv2.GaussianBlur on a float image with ksize (0, 0): round(8 sigma + 1) | 1 taps,
// normalized.
export function gaussianKernel(sigma) {
  if (sigma <= 0) return Float64Array.of(1);
  const ks = Math.round(sigma * 4 * 2 + 1) | 1;
  const r = ks >> 1;
  const k = new Float64Array(ks);
  let sum = 0;
  for (let i = 0; i < ks; i++) sum += k[i] = Math.exp(-((i - r) ** 2) / (2 * sigma * sigma));
  return k.map((v) => v / sum);
}

// The occlusion mask's Gaussian blur (px of its 256 px crop) for edge softness blur 0..1 (the edge
// mask's): a fixed 5 px at its default 0.3
export const occlusionSigma = (blur) => (5 * blur) / 0.3;

// How far (px of its N px crop) the occlusion mask grows for mask size -1..1, shrinking when negative:
// up to an eighth of the crop
export const occlusionGrow = (size, N) => Math.round((size * N) / 8);

// One axis of the edge mask: ones with zeroed edge bands, Gaussian-blurred as
// cv2.GaussianBlur does it (float kernel size rule, BORDER_REFLECT_101). The 2D mask is the outer
// product of the row and column profiles. blur 0..1; padLo / padHi: extra zeroed band in %.
export function edgeMaskProfile(n, blur = 0.3, padLo = 0, padHi = 0) {
  const amount = Math.floor(n * 0.5 * blur);
  const band = Math.max(Math.floor(amount / 2), 1);
  const f = new Float64Array(n).fill(1);
  f.fill(0, 0, Math.max(band, Math.floor((n * padLo) / 100)));
  f.fill(0, n - Math.max(band, Math.floor((n * padHi) / 100)));
  if (amount <= 0) return Float32Array.from(f);
  const k = gaussianKernel(amount * 0.25);
  const r = k.length >> 1;
  const out = new Float32Array(n);
  for (let x = 0; x < n; x++) {
    let v = 0;
    for (let i = 0; i < k.length; i++) {
      let j = Math.abs(x + i - r);
      if (j >= n) j = 2 * (n - 1) - j;
      v += f[j] * k[i];
    }
    out[x] = v;
  }
  return out;
}

// Frame region a crop of `size` pastes into (clipped to the frame), from the frame->crop map.
export function pasteBox(m, size, w, h) {
  const inv = invert(m);
  const pts = [[0, 0], [size, 0], [size, size], [0, size]].map(([x, y]) => apply(inv, x, y));
  const x1 = Math.max(0, Math.floor(Math.min(...pts.map((p) => p[0]))));
  const y1 = Math.max(0, Math.floor(Math.min(...pts.map((p) => p[1]))));
  const x2 = Math.min(w, Math.ceil(Math.max(...pts.map((p) => p[0]))));
  const y2 = Math.min(h, Math.ceil(Math.max(...pts.map((p) => p[1]))));
  return [x1, y1, x2, y2];
}

// A face's own detail, measured on its DETAIL_CROP px luma crop in the swapper's framing
// (swap_128): s, the crop's radial power spectrum (bins 0..DETAIL_CROP/2, ops.faceSpectrum),
// normalized by its power at 4..16 cycles, against the reference spectrum of sharp faces (log P a
// quadratic in log f); the face holds detail up to the first frequency from 8 cycles on where
// the ratio (5-bin moving average) falls below 1/10. -> px across the framing (cycles x 2).
// A blurred, out-of-focus or heavily compressed face measures less.
// --calibrate fits DETAIL_REF.
export const DETAIL_CROP = 256;
const DETAIL_REF = [-0.39354, -0.2615, -0.31311];
export function faceDetail(s) {
  const n = s.length;
  let norm = 0;
  for (let i = 4; i < 17; i++) norm += s[i];
  const ratio = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const lf = Math.log(i);
    ratio[i] = s[i] / norm / Math.exp((DETAIL_REF[0] * lf + DETAIL_REF[1]) * lf + DETAIL_REF[2]);
  }
  ratio[0] = s[0] / norm;
  for (let i = 8; i < n; i++) {
    let m = 0;
    for (let j = i - 2; j <= i + 2; j++) if (j < n) m += ratio[j];
    if (m / 5 < 0.1) return 2 * i;
  }
  return 2 * (n - 1);
}
