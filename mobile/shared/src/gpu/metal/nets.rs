//! The face detector (SCRFD-10G) and the face recognizer (ArcFace R50) on the GPU.

use std::collections::HashMap;

use super::conv::{buffer_with, Buffer};
use super::graph::{Act, Activation, Dense, Graph, Img8, Layer, Sampling};
use super::Device;
use crate::model::Model;
use crate::photo::{compose, invert, similarity, template, Face, Orientation, DET_SIZE, RECOGNIZE_112};

/// SCRFD-10G: a ResNet-style backbone, a feature pyramid with a bottom-up path, and a head per
/// stride (8, 16, 32).
pub(super) struct Detector {
    layers: HashMap<String, Layer>,
}

/// The backbone's stages: (stage, blocks).
const DET_STAGES: [(usize, usize); 4] = [(1, 3), (2, 4), (3, 2), (4, 3)];

impl Detector {
    pub fn load(device: &Device, model: &Model) -> Result<Detector, String> {
        let mut convs: Vec<(String, usize, usize)> = vec![("stem.0".into(), 3, 2), ("stem.1".into(), 3, 1), ("stem.2".into(), 3, 1)];
        for (stage, blocks) in DET_STAGES {
            for b in 0..blocks {
                let s = if b == 0 && stage > 1 { 2 } else { 1 };
                convs.push((format!("layer{stage}.{b}.conv1"), 3, s));
                convs.push((format!("layer{stage}.{b}.conv2"), 3, 1));
                if s == 2 {
                    convs.push((format!("layer{stage}.{b}.down"), 1, 1));
                }
            }
        }
        for i in 0..3 {
            convs.push((format!("lateral.{i}"), 1, 1));
            convs.push((format!("fpn.{i}"), 3, 1));
            for j in 0..3 {
                convs.push((format!("head.{i}.conv.{j}"), 3, 1));
            }
            convs.push((format!("head.{i}.out"), 3, 1));
        }
        for i in 0..2 {
            convs.push((format!("down.{i}"), 3, 2));
            convs.push((format!("pafpn.{i}"), 3, 1));
        }
        let mut layers = HashMap::new();
        for (name, k, s) in convs {
            let layer = Layer::load(device, model, &name, k, s, false)?;
            layers.insert(name, layer);
        }
        Ok(Detector { layers })
    }

    /// Detects faces in `img`, shown as `o` says: scales the upright image to fit the 640 px input
    /// at the top left (black elsewhere), as `photo::detector_input` does. Returns the three head
    /// outputs, each copied for the CPU as f32 [n, n, 30] for n = 80, 40, 20, and the upright
    /// image -> detector scale.
    pub fn encode(&self, g: &mut Graph, img: &Img8, o: &Orientation) -> Result<([Buffer; 3], f64), String> {
        let size = DET_SIZE as f64;
        let s = (size / o.w as f64).min(size / o.h as f64);
        // halves to even, as the reference rounds
        let nw = DET_SIZE.min((o.w as f64 * s).round_ties_even() as usize);
        let nh = DET_SIZE.min((o.h as f64 * s).round_ties_even() as usize);
        let sampling = Sampling {
            map: compose(&o.to_stored, &[1.0 / s, 0.0, 0.5 / s - 0.5, 0.0, 1.0 / s, 0.5 / s - 0.5]),
            size: (DET_SIZE, DET_SIZE),
            valid: (nw, nh),
            mul: 1.0 / 128.0,
            add: -127.5 / 128.0,
        };
        let x = g.resample(img, &sampling)?;
        let l = |name: String| self.layers.get(&name).ok_or(format!("no layer {name}"));
        let mut x = g.conv(l("stem.0".into())?, &x, 0.0)?;
        x = g.conv(l("stem.1".into())?, &x, 0.0)?;
        x = g.conv(l("stem.2".into())?, &x, 0.0)?;
        x = g.pool(&x, false)?;
        let mut feats = Vec::new();
        for (stage, blocks) in DET_STAGES {
            for b in 0..blocks {
                let p = format!("layer{stage}.{b}");
                let strided = b == 0 && stage > 1;
                let h = g.conv(l(format!("{p}.conv1"))?, &x, 0.0)?;
                let h = g.conv(l(format!("{p}.conv2"))?, &h, 1.0)?;
                let shortcut = if strided {
                    let pooled = g.pool(&x, true)?;
                    g.conv(l(format!("{p}.down"))?, &pooled, 1.0)?
                } else {
                    x.clone()
                };
                x = g.affine(&h, None, Some(&shortcut), Activation::Relu)?;
            }
            if stage > 1 {
                feats.push(x.clone());
            }
        }
        let lat: Vec<Act> = (0..3).map(|i| g.conv(l(format!("lateral.{i}"))?, &feats[i], 1.0)).collect::<Result<_, _>>()?;
        let l2 = lat[2].clone();
        let l1 = g.up_add(&lat[1], &l2)?;
        let l0 = g.up_add(&lat[0], &l1)?;
        let p3 = g.conv(l("fpn.0".into())?, &l0, 1.0)?;
        let p4 = g.conv(l("fpn.1".into())?, &l1, 1.0)?;
        let p5 = g.conv(l("fpn.2".into())?, &l2, 1.0)?;
        let d = g.conv(l("down.0".into())?, &p3, 1.0)?;
        let p4 = g.affine(&p4, None, Some(&d), Activation::None)?;
        let d = g.conv(l("down.1".into())?, &p4, 1.0)?;
        let p5 = g.affine(&p5, None, Some(&d), Activation::None)?;
        let p4 = g.conv(l("pafpn.0".into())?, &p4, 1.0)?;
        let p5 = g.conv(l("pafpn.1".into())?, &p5, 1.0)?;
        let mut heads = Vec::new();
        for (i, f) in [p3, p4, p5].into_iter().enumerate() {
            let mut f = f;
            for j in 0..3 {
                f = g.conv(l(format!("head.{i}.conv.{j}"))?, &f, 0.0)?;
            }
            let out = g.conv(l(format!("head.{i}.out"))?, &f, 1.0)?;
            heads.push(g.to_host(&out)?);
        }
        let heads: [Buffer; 3] = heads.try_into().map_err(|_| "three heads")?;
        Ok((heads, s))
    }
}

/// ArcFace R50 (an IResNet): four stages of pre-norm residual blocks with PReLU, then a fully
/// connected layer to a 512-value embedding.
pub(super) struct Recognizer {
    layers: HashMap<String, Layer>,
    /// per-channel buffers: batch-norm scales and shifts, PReLU slopes
    params: HashMap<String, Buffer>,
    fc: Dense,
}

const REC_STAGES: [usize; 4] = [3, 4, 14, 3];

impl Recognizer {
    pub fn load(device: &Device, model: &Model) -> Result<Recognizer, String> {
        let mut layers = HashMap::new();
        let mut params = HashMap::new();
        let mut vector = |name: String| -> Result<(), String> {
            let v = model.tensor(&name)?.values;
            params.insert(name, buffer_with(device, &v)?);
            Ok(())
        };
        layers.insert("conv0".to_string(), Layer::load(device, model, "conv0", 3, 1, false)?);
        vector("prelu0".into())?;
        for (stage, blocks) in REC_STAGES.into_iter().enumerate() {
            for b in 0..blocks {
                let p = format!("layers.{stage}.{b}");
                let s = if b == 0 { 2 } else { 1 };
                vector(format!("{p}.bn1.scale"))?;
                vector(format!("{p}.bn1.shift"))?;
                vector(format!("{p}.prelu"))?;
                layers.insert(format!("{p}.conv1"), Layer::load(device, model, &format!("{p}.conv1"), 3, 1, false)?);
                layers.insert(format!("{p}.conv2"), Layer::load(device, model, &format!("{p}.conv2"), 3, s, false)?);
                if b == 0 {
                    layers.insert(format!("{p}.down"), Layer::load(device, model, &format!("{p}.down"), 1, 2, false)?);
                }
            }
        }
        let fc = Dense::new(device, &model.weight("fc")?.values, &model.tensor("fc.bias")?.values)?;
        Ok(Recognizer { layers, params, fc })
    }

    /// The embedding (not normalized) of `face` (upright coordinates) in `img`, f32 [512] in
    /// shared memory: the face's 112 px aligned crop in [-1, 1], as `photo::recognizer_input` makes
    /// it, through the network.
    pub fn encode(&self, g: &mut Graph, img: &Img8, o: &Orientation, face: &Face) -> Result<Buffer, String> {
        let m = similarity(&face.kps, &template(&RECOGNIZE_112, 112.0));
        let map = compose(&o.to_stored, &invert(&m));
        let sampling = Sampling { map, size: (112, 112), valid: (112, 112), mul: 1.0 / 127.5, add: -1.0 };
        let x = g.resample(img, &sampling)?;
        let l = |name: String| self.layers.get(&name).ok_or(format!("no layer {name}"));
        let v = |name: String| self.params.get(&name).ok_or(format!("no parameter {name}"));
        let x = g.conv(l("conv0".into())?, &x, 1.0)?;
        let mut x = g.affine(&x, None, None, Activation::Prelu(v("prelu0".into())?))?;
        for (stage, blocks) in REC_STAGES.into_iter().enumerate() {
            for b in 0..blocks {
                let p = format!("layers.{stage}.{b}");
                let h = g.affine(&x, Some((v(format!("{p}.bn1.scale"))?, v(format!("{p}.bn1.shift"))?)), None, Activation::None)?;
                let h = g.conv(l(format!("{p}.conv1"))?, &h, 1.0)?;
                let h = g.affine(&h, None, None, Activation::Prelu(v(format!("{p}.prelu"))?))?;
                let h = g.conv(l(format!("{p}.conv2"))?, &h, 1.0)?;
                let shortcut = if b == 0 { g.conv(l(format!("{p}.down"))?, &x, 1.0)? } else { x.clone() };
                x = g.affine(&h, None, Some(&shortcut), Activation::None)?;
            }
        }
        // NHWC flattened (affine leaves it compact), as the fully connected layer was laid out
        if x.h * x.w * x.c != self.fc.cols {
            return Err(format!("fc takes {} values, the backbone gives {}", self.fc.cols, x.h * x.w * x.c));
        }
        g.dense(&self.fc, (&x.buf, 0), true)
    }
}
