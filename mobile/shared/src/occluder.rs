//! The occlusion mask (the community-trained XSeg): what of a face crop is face, so hands, glasses
//! and hair in front of it stay. Its weights as the GPU takes them, and the settings of
//! the mask made from its output.
//!
//!   encoder     6 levels of 3x3 convs (32 .. 256 channels), each level then blurred and halved
//!               ([1,3,3,1] and [1,2,1] per axis, then 2x2 averages)
//!   bottleneck  dense 4096 -> 512 -> 4096 on the 4 px map
//!   decoder     per level the stride-2 transposed conv (polyphase), joined with the encoder's map
//!               of that size, then 3x3 convs; a 3x3 conv to the mask, sigmoid
//!
//! Every conv but the last is followed by FRN + TLU.

use crate::cpu::{ConvLayer, Dense};
use crate::model::Model;

pub const SIZE: usize = 256;
pub const ENCODER: [&[&str]; 6] = [&["conv01", "conv02"], &["conv11", "conv12"], &["conv21", "conv22"], &["conv31", "conv32", "conv33"], &["conv41", "conv42", "conv43"], &["conv51", "conv52", "conv53"]];
pub const DECODER: [(&str, &[&str]); 6] = [
    ("up5", &["uconv53", "uconv52", "uconv51"]),
    ("up4", &["uconv43", "uconv42", "uconv41"]),
    ("up3", &["uconv33", "uconv32", "uconv31"]),
    ("up2", &["uconv22", "uconv21"]),
    ("up1", &["uconv12", "uconv11"]),
    ("up0", &["uconv02", "uconv01"]),
];
/// the downsampling blur of encoder levels 0 and 1 (per axis); the deeper ones are 2x2 averages
pub const BLURS: [&[f32]; 2] = [&[1.0, 3.0, 3.0, 1.0], &[1.0, 2.0, 1.0]];

/// A conv with its FRN: [gamma, beta, tau] (3 cout) and epsilon.
pub struct FrnConv {
    pub conv: ConvLayer,
    pub frn: Vec<f32>,
    pub eps: f32,
}

impl FrnConv {
    fn load(model: &Model, name: &str, up: bool) -> Result<FrnConv, String> {
        let conv = if up { ConvLayer::polyphase(model, name, 0)? } else { ConvLayer::load(model, name, 3, 1, [1, 1], false)? };
        let frn = model.tensor(&format!("{name}.frn"))?.values;
        let eps = model.tensor(&format!("{name}.eps"))?.values[0];
        Ok(FrnConv { conv, frn, eps })
    }
}

pub struct Occluder {
    pub encoder: Vec<Vec<FrnConv>>,
    /// dense1's columns reordered to take the 4 px map as it is stored (NHWC); the model's take
    /// its NCHW flatten
    pub dense1: Dense,
    /// its rows are NHWC already
    pub dense2: Dense,
    /// per level, the transposed conv then the 3x3 convs
    pub decoder: Vec<(FrnConv, Vec<FrnConv>)>,
    pub out: ConvLayer,
}

impl Occluder {
    pub fn load(model: &Model) -> Result<Occluder, String> {
        let encoder = ENCODER.iter().map(|names| names.iter().map(|n| FrnConv::load(model, n, false)).collect()).collect::<Result<Vec<_>, _>>()?;
        let decoder = DECODER
            .iter()
            .map(|(up, names)| Ok((FrnConv::load(model, up, true)?, names.iter().map(|n| FrnConv::load(model, n, false)).collect::<Result<Vec<_>, String>>()?)))
            .collect::<Result<Vec<_>, String>>()?;
        let mut dense1 = Dense::load(model, "dense1")?;
        let c = dense1.cols / 16;
        let mut w = vec![0f32; dense1.w.len()];
        for r in 0..dense1.rows {
            for ch in 0..c {
                for p in 0..16 {
                    w[r * dense1.cols + p * c + ch] = dense1.w[r * dense1.cols + ch * 16 + p];
                }
            }
        }
        dense1.w = w;
        Ok(Occluder { encoder, dense1, dense2: Dense::load(model, "dense2")?, decoder, out: ConvLayer::load(model, "out", 3, 1, [1, 1], false)? })
    }

}

/// The Gaussian blur of the occlusion mask, in px of its 256 px crop, for the edge softness
/// (Edges, 0..1): 5 px at its default 0.3.
pub fn sigma(edges: f64) -> f64 {
    5.0 * edges / 0.3
}

/// How far the mask grows (px of its n px crop; it shrinks when negative) for Coverage -1..1:
/// up to an eighth of the crop.
pub fn grow(coverage: f64, n: usize) -> isize {
    (coverage * n as f64 / 8.0).round() as isize
}

/// OpenCV's Gaussian kernel for a float image: round(8 sigma + 1) | 1 taps, normalized.
pub fn gaussian(sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let ks = ((sigma * 8.0 + 1.0).round() as usize) | 1;
    let r = (ks / 2) as f64;
    let k: Vec<f64> = (0..ks).map(|i| (-((i as f64 - r).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = k.iter().sum();
    k.iter().map(|v| (v / sum) as f32).collect()
}
