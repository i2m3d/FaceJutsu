//! The face enhancer (GPEN-BFR, 256 or 512 px) and how its result goes into the frame: its
//! weights as the GPU takes them.
//!
//!   encoder  a 1x1 conv, then per level the [1, 3, 3, 1] blur and a stride-2 3x3 conv, to 4 px
//!   style    a linear on the 4 px map, pixel norm, 8 linears -> w; every layer's style from w in
//!            one linear, and each modulated conv's demodulation s^2 @ demod^T
//!   decoder  from a learned 4 px input, per resolution: the stride-2 transposed conv (polyphase,
//!            blurred in its activation), then a 3x3 conv, both modulated (the input scaled by the
//!            style) and demodulated, each followed by the encoder's map of that size times a
//!            scalar; a 1x1 conv to RGB per resolution, added to the one before upsampled.

use crate::cpu::{ConvLayer, Dense};
use crate::model::Model;
use crate::photo::{similarity, SWAP_128};

/// The wider framing of the enhancer that sees the whole head (GPEN-512).
pub const WIDE_512: [[f64; 2]; 5] = [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904], [0.39308822, 0.725411], [0.61150205, 0.72490465]];

/// The detail a swap at pixelshuffle 1 to 4 holds, px across the swapper's framing; a face with
/// less detail of its own gives less: softmin(a * its detail, this), DETAIL_FIT = (a, p).
const SWAP_DETAIL: [f64; 4] = [70.0, 108.0, 124.0, 128.0];
const DETAIL_FIT: (f64, f64) = (0.944, 2.5);

pub fn swap_detail(pixelshuffle: usize, face_detail: Option<f64>) -> f64 {
    let most = SWAP_DETAIL[pixelshuffle.clamp(1, 4) - 1];
    match face_detail {
        None => most,
        Some(d) => {
            let (a, p) = DETAIL_FIT;
            (((a * d).powf(-p) + most.powf(-p)).powf(-1.0 / p)).round()
        }
    }
}

/// The px of an enhancer's crop below which a swap with `detail` px (swap_detail) has all of it:
/// GPEN-256 shares the swapper's framing; GPEN-512's wider one holds 128 px in 143.
pub fn band(detail: f64, template: &[[f64; 2]; 5]) -> usize {
    let m = similarity(template, &SWAP_128);
    (detail * m[0].hypot(m[3])).round() as usize
}

pub enum Kind {
    Conv,
    Up,
    Rgb,
}

/// A decoder layer: a modulated conv (3x3, or the polyphase transposed one) with its activation,
/// or a modulated 1x1 conv to RGB.
pub struct Styled {
    pub kind: Kind,
    /// where its style starts in the row of all styles, and how long it is (its input channels)
    pub off: usize,
    pub cin: usize,
    pub conv: Option<ConvLayer>,
    /// [cout, cin]
    pub demod: Option<Dense>,
    /// [2 cout]: the bias of [conv | noise * feat]
    pub act: Vec<f32>,
    pub noise: f32,
    /// the encoder map it is joined with
    pub feat: usize,
    /// to RGB: [3, cin] and [3]
    pub rgb: Vec<f32>,
    pub rgb_bias: Vec<f32>,
}

pub struct Enhancer {
    /// the crop's size
    pub size: usize,
    pub levels: usize,
    /// enc.0 (1x1), then 3x3 stride 2 without padding (after the blur)
    pub enc: Vec<ConvLayer>,
    pub fc: Dense,
    pub mapping: Vec<Dense>,
    pub modulation: Dense,
    /// the learned 4 px input, [4, 4, c]
    pub konst: Vec<f32>,
    pub layers: Vec<Styled>,
}

impl Enhancer {
    pub fn load(model: &Model) -> Result<Enhancer, String> {
        let mut levels = 0;
        while model.has(&format!("enc.{levels}.weight")) {
            levels += 1;
        }
        if levels < 2 {
            return Err("not an enhancer: no encoder".into());
        }
        let mut enc = vec![ConvLayer::load(model, "enc.0", 1, 1, [0, 0], false)?];
        for i in 1..levels {
            enc.push(ConvLayer::load(model, &format!("enc.{i}"), 3, 2, [0, 0], false)?);
        }
        let mapping = (0..8).map(|j| Dense::load(model, &format!("style.{j}"))).collect::<Result<Vec<_>, _>>()?;
        let mut names = vec![("conv.0".to_string(), Kind::Conv, levels - 1), ("rgb.0".to_string(), Kind::Rgb, 0)];
        for i in 0..levels - 1 {
            let f = levels - 2 - i;
            names.push((format!("up.{i}"), Kind::Up, f));
            names.push((format!("conv.{}", i + 1), Kind::Conv, f));
            names.push((format!("rgb.{}", i + 1), Kind::Rgb, 0));
        }
        let mut layers = Vec::new();
        let mut off = 0;
        for (name, kind, feat) in names {
            let layer = match kind {
                Kind::Rgb => {
                    let rgb = model.tensor(&format!("{name}.weight"))?.values;
                    let cin = rgb.len() / 3;
                    Styled { kind, off, cin, conv: None, demod: None, act: Vec::new(), noise: 0.0, feat, rgb, rgb_bias: model.tensor(&format!("{name}.bias"))?.values }
                }
                Kind::Conv | Kind::Up => {
                    let conv = match kind {
                        Kind::Up => ConvLayer::polyphase(model, &name, 1)?,
                        _ => ConvLayer::load(model, &name, 3, 1, [1, 1], false)?,
                    };
                    let demod = Dense::load(model, &format!("{name}.demod"))?;
                    let act = model.tensor(&format!("{name}.act"))?.values;
                    let noise = model.tensor(&format!("{name}.noise"))?.values[0];
                    Styled { kind, off, cin: conv.cin, conv: Some(conv), demod: Some(demod), act, noise, feat, rgb: Vec::new(), rgb_bias: Vec::new() }
                }
            };
            off += layer.cin;
            layers.push(layer);
        }
        let modulation = Dense::load(model, "mod")?;
        if off != modulation.rows {
            return Err(format!("the enhancer's layers take {off} style values of {}", modulation.rows));
        }
        Ok(Enhancer { size: 1 << (levels + 1), levels, enc, fc: Dense::load(model, "fc")?, mapping, modulation, konst: model.tensor("const")?.values, layers })
    }

}
