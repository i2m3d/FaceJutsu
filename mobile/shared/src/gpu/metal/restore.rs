//! The face enhancer (GPEN) and the occlusion mask (XSeg) on the GPU, step for step as the CPU
//! runs them (crate::enhancer, crate::occluder), whose loaders prepare the weights.

use super::conv::{buffer_with, Buffer};
use super::graph::{Act, Dense, Finish, Graph, Layer};
use super::Device;
use crate::enhancer::{self, Kind};
use crate::model::Model;
use crate::occluder;

struct Styled {
    kind: Kind,
    /// where its style starts in the row of all styles
    off: usize,
    conv: Option<Layer>,
    demod: Option<Dense>,
    act: Option<Buffer>,
    noise: f32,
    feat: usize,
    rgb: Option<(Buffer, Buffer)>,
}

pub(super) struct Enhancer {
    pub size: usize,
    levels: usize,
    enc: Vec<Layer>,
    fc: Dense,
    mapping: Vec<Dense>,
    modulation: Dense,
    /// the learned 4 px input
    konst: Act,
    layers: Vec<Styled>,
}

impl Enhancer {
    pub fn load(device: &Device, model: &Model) -> Result<Enhancer, String> {
        let e = enhancer::Enhancer::load(model)?;
        let enc = e.enc.iter().map(|l| Layer::from_cpu(device, l)).collect::<Result<_, _>>()?;
        let mapping = e.mapping.iter().map(|d| Dense::from_cpu(device, d)).collect::<Result<_, _>>()?;
        let c = e.konst.len() / 16;
        let layers = e
            .layers
            .into_iter()
            .map(|l| {
                Ok(Styled {
                    off: l.off,
                    conv: l.conv.as_ref().map(|c| Layer::from_cpu(device, c)).transpose()?,
                    demod: l.demod.as_ref().map(|d| Dense::from_cpu(device, d)).transpose()?,
                    act: if l.act.is_empty() { None } else { Some(buffer_with(device, &l.act)?) },
                    noise: l.noise,
                    feat: l.feat,
                    rgb: if l.rgb.is_empty() { None } else { Some((buffer_with(device, &l.rgb)?, buffer_with(device, &l.rgb_bias)?)) },
                    kind: l.kind,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Enhancer {
            size: e.size,
            levels: e.levels,
            enc,
            fc: Dense::from_cpu(device, &e.fc)?,
            mapping,
            modulation: Dense::from_cpu(device, &e.modulation)?,
            konst: Act::upload(device, 4, 4, c, &e.konst)?,
            layers,
        })
    }

    /// x: the crop, [size, size, 3] in [-1, 1] -> f32 [size, size, 3], unclamped.
    pub fn encode(&self, g: &mut Graph, x: &Act) -> Result<Buffer, String> {
        let mut feats = vec![g.conv(&self.enc[0], x, 0.2)?];
        for i in 1..self.levels {
            let blurred = g.fir_down(&feats[i - 1])?;
            feats.push(g.conv(&self.enc[i], &blurred, 0.2)?);
        }
        let last = &feats[self.levels - 1];
        if last.sw != last.w || last.h * last.w * last.c != self.fc.cols {
            return Err(format!("the enhancer's linear takes {} values, its encoder gives {}x{}x{}", self.fc.cols, last.h, last.w, last.c));
        }
        let mut z = g.dense_act(&self.fc, (&last.buf, 0), true, 0.2)?;
        g.pixel_norm(&z, self.fc.rows)?;
        for d in &self.mapping {
            z = g.dense_act(d, (&z, 0), false, 0.2)?;
        }
        let styles = g.dense(&self.modulation, (&z, 0), false)?;
        let sq = g.square(&styles, self.modulation.rows)?;
        let mut h = self.konst.clone();
        let mut rgb: Option<Buffer> = None;
        for l in &self.layers {
            let style = (&styles, l.off * 4);
            match l.kind {
                Kind::Rgb => {
                    let (wt, bias) = l.rgb.as_ref().ok_or("an RGB layer without weights")?;
                    rgb = Some(g.to_rgb(&h, style, wt, bias, rgb.as_ref())?);
                }
                Kind::Conv | Kind::Up => {
                    let y = g.conv_scaled(l.conv.as_ref().unwrap(), &h, 1.0, Some(style))?;
                    let d = g.dense(l.demod.as_ref().unwrap(), (&sq, l.off * 4), false)?;
                    h = g.styled_act(&y, &d, l.act.as_ref().unwrap(), l.noise, &feats[l.feat], matches!(l.kind, Kind::Up))?;
                }
            }
        }
        rgb.ok_or_else(|| "the enhancer made no image".into())
    }
}

struct FrnConv {
    conv: Layer,
    frn: Buffer,
    eps: f32,
    c: usize,
}

impl FrnConv {
    fn from_cpu(device: &Device, l: &occluder::FrnConv) -> Result<FrnConv, String> {
        Ok(FrnConv { conv: Layer::from_cpu(device, &l.conv)?, frn: buffer_with(device, &l.frn)?, eps: l.eps, c: l.frn.len() / 3 })
    }

    /// The conv through its FRN + TLU.
    fn run(&self, g: &mut Graph, x: &Act) -> Result<Act, String> {
        let y = g.conv(&self.conv, x, 1.0)?;
        let pro = g.frn_stats(&y, &self.frn, self.eps, 1)?;
        g.frn_out(&y, &pro, self.c, false, None)
    }
}

pub(super) struct Occluder {
    encoder: Vec<Vec<FrnConv>>,
    dense1: Dense,
    dense2: Dense,
    decoder: Vec<(FrnConv, Vec<FrnConv>)>,
    out: Layer,
}

impl Occluder {
    pub fn load(device: &Device, model: &Model) -> Result<Occluder, String> {
        let o = occluder::Occluder::load(model)?;
        let level = |ls: &Vec<occluder::FrnConv>| ls.iter().map(|l| FrnConv::from_cpu(device, l)).collect::<Result<Vec<_>, _>>();
        Ok(Occluder {
            encoder: o.encoder.iter().map(level).collect::<Result<_, _>>()?,
            dense1: Dense::from_cpu(device, &o.dense1)?,
            dense2: Dense::from_cpu(device, &o.dense2)?,
            decoder: o.decoder.iter().map(|(up, ls)| Ok((FrnConv::from_cpu(device, up)?, level(ls)?))).collect::<Result<_, String>>()?,
            out: Layer::from_cpu(device, &o.out)?,
        })
    }

    /// x: the crop, [256, 256, 3] in [0, 1] -> XSeg's output, f32 [256, 256] in 0..1 (1: face).
    pub fn encode(&self, g: &mut Graph, x: &Act) -> Result<Buffer, String> {
        let mut h = x.clone();
        let mut skips = Vec::new();
        for (i, level) in self.encoder.iter().enumerate() {
            for l in level {
                h = l.run(g, &h)?;
            }
            skips.push(h.clone());
            h = match occluder::BLURS.get(i) {
                Some(t) => g.blur_down(&h, t)?,
                None => g.pool(&h, true)?,
            };
        }
        if h.sw != h.w || h.h * h.w * h.c != self.dense1.cols {
            return Err(format!("XSeg's bottleneck takes {} values, its encoder gives {}x{}x{}", self.dense1.cols, h.h, h.w, h.c));
        }
        let z = g.dense(&self.dense1, (&h.buf, 0), true)?;
        let z = g.dense(&self.dense2, (&z, 0), false)?;
        h = g.to_half(&z, 4, 4, self.dense2.rows / 16)?;
        for (up, convs) in &self.decoder {
            let u = g.conv(&up.conv, &h, 1.0)?;
            let pro = g.frn_stats(&u, &up.frn, up.eps, 4)?;
            let skip = skips.pop().ok_or("XSeg: no encoder map left")?;
            h = g.frn_out(&u, &pro, up.c, true, Some(&skip))?;
            for l in convs {
                h = l.run(g, &h)?;
            }
        }
        let y = g.conv(&self.out, &h, 1.0)?;
        let out = g.f32_buffer(y.h * y.w)?;
        g.finish_into(&y, Finish::Sigmoid, (&out, 0))?;
        Ok(out)
    }
}

/// The occlusion mask's Gaussian taps for Edges `edges`, on the GPU, and their radius.
pub(super) fn mask_taps(device: &Device, edges: f64) -> Result<(Buffer, usize), String> {
    let taps = occluder::gaussian(occluder::sigma(edges));
    Ok((buffer_with(device, &taps)?, taps.len() / 2))
}
