//! The CPU side of a photo swap: preparing the detector input, reading faces out of the
//! detector's outputs, aligning face crops, and pasting swapped faces back. Images are f32 HWC
//! RGB, 0..255 unless noted.

/// The detector's input size (square).
pub const DET_SIZE: usize = 640;
pub const DET_SCORE: f64 = 0.5;
pub const DET_NMS: f64 = 0.4;

/// 5-point templates (left eye, right eye, nose, left and right mouth corners), as fractions of
/// the crop size: ArcFace's 112 px crop and the swapper's 128 px crop.
pub const RECOGNIZE_112: [[f64; 2]; 5] = [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.50022500, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]];
pub const SWAP_128: [[f64; 2]; 5] = [[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219], [0.38710391, 0.72160547], [0.61507734, 0.72034453]];

#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub data: Vec<f32>,
}

/// A 2x3 affine map, row-major: (x, y) -> (m[0] x + m[1] y + m[2], m[3] x + m[4] y + m[5]).
pub type Affine = [f64; 6];

#[derive(Clone, Debug)]
pub struct Face {
    pub bbox: [f64; 4],
    pub score: f64,
    pub kps: [[f64; 2]; 5],
}

/// The least-squares similarity transform (rotation, uniform scale, translation) taking `src`
/// points onto `dst` points.
pub fn similarity(src: &[[f64; 2]; 5], dst: &[[f64; 2]; 5]) -> Affine {
    let mean = |p: &[[f64; 2]; 5]| [p.iter().map(|q| q[0]).sum::<f64>() / 5.0, p.iter().map(|q| q[1]).sum::<f64>() / 5.0];
    let (ms, md) = (mean(src), mean(dst));
    let (mut n, mut a, mut b) = (0.0, 0.0, 0.0);
    for (s, d) in src.iter().zip(dst) {
        let (sx, sy, dx, dy) = (s[0] - ms[0], s[1] - ms[1], d[0] - md[0], d[1] - md[1]);
        n += sx * sx + sy * sy;
        a += sx * dx + sy * dy;
        b += sx * dy - sy * dx;
    }
    let (a, b) = (a / n, b / n);
    [a, -b, md[0] - (a * ms[0] - b * ms[1]), b, a, md[1] - (b * ms[0] + a * ms[1])]
}

pub fn invert(m: &Affine) -> Affine {
    let det = m[0] * m[4] - m[1] * m[3];
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])]
}

/// a after b: the map taking p to a(b(p)).
pub fn compose(a: &Affine, b: &Affine) -> Affine {
    [
        a[0] * b[0] + a[1] * b[3],
        a[0] * b[1] + a[1] * b[4],
        a[0] * b[2] + a[1] * b[5] + a[2],
        a[3] * b[0] + a[4] * b[3],
        a[3] * b[1] + a[4] * b[4],
        a[3] * b[2] + a[4] * b[5] + a[5],
    ]
}

/// How a stored frame is shown: turned `rotation` degrees clockwise (0, 90, 180 or 270), as a
/// video's display matrix says. Faces are found and aligned on the upright frame; the frame itself
/// stays as stored, and every map into it goes through `to_stored`.
#[derive(Clone, Copy, Debug)]
pub struct Orientation {
    /// upright pixel -> stored pixel (pixel centers at integers)
    pub to_stored: Affine,
    /// the upright frame's size
    pub w: usize,
    pub h: usize,
}

impl Orientation {
    pub fn upright(w: usize, h: usize) -> Orientation {
        Orientation { to_stored: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], w, h }
    }

    /// For a frame stored `w` x `h` and shown turned `rotation` degrees clockwise.
    pub fn new(rotation: u32, w: usize, h: usize) -> Result<Orientation, String> {
        let (wl, hl) = ((w - 1) as f64, (h - 1) as f64);
        Ok(match rotation % 360 {
            0 => Orientation::upright(w, h),
            // stored (x, y) shows at (h - 1 - y, x)
            90 => Orientation { to_stored: [0.0, 1.0, 0.0, -1.0, 0.0, hl], w: h, h: w },
            180 => Orientation { to_stored: [-1.0, 0.0, wl, 0.0, -1.0, hl], w, h },
            // stored (x, y) shows at (y, w - 1 - x)
            270 => Orientation { to_stored: [0.0, -1.0, wl, 1.0, 0.0, 0.0], w: h, h: w },
            r => return Err(format!("a rotation of {r} degrees")),
        })
    }
}

/// The template points for a crop of `size` px.
pub fn template(t: &[[f64; 2]; 5], size: f64) -> [[f64; 2]; 5] {
    t.map(|p| [p[0] * size, p[1] * size])
}

/// Faces from the detector's three head outputs (strides 8, 16, 32; each [n, n, 30] NHWC:
/// 2 class logits, 2 x 4 box distances, 2 x 10 keypoint offsets, in strides), in image
/// coordinates, after non-maximum suppression, best first.
pub fn decode_detections(outs: [&[f32]; 3], s: f64) -> Vec<Face> {
    let mut found: Vec<Face> = Vec::new();
    // in the reference's order (stride, anchor, cell), so ties sort the same
    for (i, stride) in [8usize, 16, 32].into_iter().enumerate() {
        let n = DET_SIZE / stride;
        for a in 0..2 {
            for cell in 0..n * n {
                let o = &outs[i][cell * 30..][..30];
                let (cx, cy) = (((cell % n) * stride) as f64, ((cell / n) * stride) as f64);
                let score = 1.0 / (1.0 + (-(o[a] as f64)).exp());
                if score < DET_SCORE {
                    continue;
                }
                let r = |j: usize| o[2 + 4 * a + j] as f64 * stride as f64;
                let bbox = [(cx - r(0)) / s, (cy - r(1)) / s, (cx + r(2)) / s, (cy + r(3)) / s];
                let kps = std::array::from_fn(|p| {
                    let k = |j: usize| o[10 + 10 * a + 2 * p + j] as f64 * stride as f64;
                    [(k(0) + cx) / s, (k(1) + cy) / s]
                });
                found.push(Face { bbox, score, kps });
            }
        }
    }
    // stable: equal scores keep their order
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for f in found {
        let overlaps = kept.iter().any(|k| {
            let ix = (f.bbox[2].min(k.bbox[2]) - f.bbox[0].max(k.bbox[0])).max(0.0);
            let iy = (f.bbox[3].min(k.bbox[3]) - f.bbox[1].max(k.bbox[1])).max(0.0);
            let inter = ix * iy;
            let area = |b: &[f64; 4]| (b[2] - b[0]) * (b[3] - b[1]);
            inter / (area(&f.bbox) + area(&k.bbox) - inter).max(1e-9) > DET_NMS
        });
        if !overlaps {
            kept.push(f);
        }
    }
    kept
}

/// The identity latent the swapper takes: the embedding normalized, mapped through emap
/// ([512, 512], row-major, e @ emap), normalized again.
pub fn latent(embedding: &[f32], emap: &[f32]) -> Vec<f32> {
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let e: Vec<f64> = embedding.iter().map(|v| *v as f64).collect();
    let ne = norm(&e);
    let n = e.len();
    let lat: Vec<f64> = (0..n).map(|j| (0..n).map(|i| e[i] / ne * emap[i * n + j] as f64).sum()).collect();
    let nl = norm(&lat);
    lat.iter().map(|v| (v / nl) as f32).collect()
}

/// One axis of the paste-back edge mask: ones with zeroed edge bands, Gaussian-blurred (OpenCV's
/// kernel size rule, reflect-101 borders). The 2D mask is the outer product of two of these.
pub fn edge_mask_profile(n: usize, blur: f64) -> Vec<f32> {
    let amount = (n as f64 * 0.5 * blur) as usize;
    let area = (amount / 2).max(1);
    let mut f = vec![1.0f64; n];
    f[..area].iter_mut().for_each(|v| *v = 0.0);
    f[n - area..].iter_mut().for_each(|v| *v = 0.0);
    if amount == 0 {
        return f.iter().map(|v| *v as f32).collect();
    }
    let sigma = amount as f64 * 0.25;
    let ks = ((sigma * 4.0 * 2.0 + 1.0).round() as usize) | 1;
    let r = ks / 2;
    let mut k: Vec<f64> = (0..ks).map(|i| (-((i as f64 - r as f64).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    (0..n)
        .map(|i| {
            (0..ks)
                .map(|j| {
                    let idx = (i as isize + j as isize - r as isize).unsigned_abs();
                    let idx = if idx >= n { 2 * (n - 1) - idx } else { idx };
                    f[idx] * k[j]
                })
                .sum::<f64>() as f32
        })
        .collect()
}

/// The box [x1, y1, x2) x [y1, y2) of a w x h frame that an s px crop covers, from its frame ->
/// crop map.
pub fn paste_box(m_frame_to_crop: &Affine, s: usize, w: usize, h: usize) -> [usize; 4] {
    let s = s as f64;
    let inv = invert(m_frame_to_crop);
    let corners = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)].map(|(x, y)| (inv[0] * x + inv[1] * y + inv[2], inv[3] * x + inv[4] * y + inv[5]));
    let clampi = |v: f64, hi: usize| (v as isize).clamp(0, hi as isize) as usize;
    [
        clampi(corners.iter().map(|c| c.0).fold(f64::MAX, f64::min).floor(), w),
        clampi(corners.iter().map(|c| c.1).fold(f64::MAX, f64::min).floor(), h),
        clampi(corners.iter().map(|c| c.0).fold(f64::MIN, f64::max).ceil(), w),
        clampi(corners.iter().map(|c| c.1).fold(f64::MIN, f64::max).ceil(), h),
    ]
}

/// The crop a face's own detail is measured on, px.
pub const DETAIL_CROP: usize = 256;
/// The spectrum of sharp faces: log P = a (log f)^2 + b log f + c.
const DETAIL_REF: [f64; 3] = [-0.39354, -0.2615, -0.31311];

/// In-place radix-2 FFT of (re, im) pairs, n a power of 2.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

/// A face's spectrum, for `face_detail`: the power of its DETAIL_CROP px crop (the swapper's
/// framing, from a 0..255 crop), luma in 0..1 less 1/2, Hann-windowed in x and y, summed per
/// radial frequency round(|f|) (bins 0..n/2, beyond: the last).
pub fn face_spectrum(crop: &Image) -> Vec<f64> {
    let n = crop.w;
    let hann: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()).collect();
    let (mut re, mut im) = (vec![0f64; n * n], vec![0f64; n * n]);
    for y in 0..n {
        for x in 0..n {
            let p = &crop.data[(y * n + x) * 3..][..3];
            let l = (0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64) / 255.0 - 0.5;
            re[y * n + x] = l * hann[x] * hann[y];
        }
    }
    for y in 0..n {
        fft(&mut re[y * n..][..n], &mut im[y * n..][..n]);
    }
    let (mut cr, mut ci) = (vec![0f64; n], vec![0f64; n]);
    let mut bins = vec![0f64; n / 2 + 1];
    let freq = |i: usize| if i < n.div_ceil(2) { i as f64 } else { i as f64 - n as f64 };
    for x in 0..n {
        for y in 0..n {
            cr[y] = re[y * n + x];
            ci[y] = im[y * n + x];
        }
        fft(&mut cr, &mut ci);
        for y in 0..n {
            let r = (freq(x).powi(2) + freq(y).powi(2)).sqrt().round_ties_even() as usize;
            bins[r.min(n / 2)] += cr[y] * cr[y] + ci[y] * ci[y];
        }
    }
    bins
}

/// The px of detail (across the swapper's framing) a face with spectrum `s` holds: s normalized
/// by its power at 4..16 cycles, against the sharp faces' reference; up to the first frequency
/// from 8 cycles on where the ratio (5-bin moving average) falls below 1/10.
pub fn face_detail(s: &[f64]) -> f64 {
    let n = s.len();
    let norm: f64 = s[4..17].iter().sum();
    let mut ratio = vec![s[0] / norm; n];
    for (i, r) in ratio.iter_mut().enumerate().skip(1) {
        let lf = (i as f64).ln();
        *r = s[i] / norm / ((DETAIL_REF[0] * lf + DETAIL_REF[1]) * lf + DETAIL_REF[2]).exp();
    }
    for i in 8..n {
        let m: f64 = (i - 2..=i + 2).filter(|j| *j < n).map(|j| ratio[j]).sum();
        if m / 5.0 < 0.1 {
            return 2.0 * i as f64;
        }
    }
    2.0 * (n - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_applies_the_right_map_first() {
        let a = [2.0, 0.5, 1.0, -1.0, 3.0, 4.0];
        let b = [0.0, 1.0, -2.0, 1.5, 0.0, 0.5];
        let apply = |m: &Affine, (x, y): (f64, f64)| (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
        let p = (3.0, -2.0);
        let (want, got) = (apply(&a, apply(&b, p)), apply(&compose(&a, &b), p));
        assert!((want.0 - got.0).abs() < 1e-12 && (want.1 - got.1).abs() < 1e-12);
        let id = compose(&a, &invert(&a));
        assert!(id.iter().zip([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).all(|(g, w)| (g - w).abs() < 1e-12));
    }
}
