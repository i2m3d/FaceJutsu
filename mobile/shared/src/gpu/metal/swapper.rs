//! The swapper (inswapper_128) on the GPU: the target face crop and the source identity's latent
//! in, the swapped face out. The style vectors (two per residual block) come from the latent
//! through one fully connected layer holding all twelve style layers.

use std::collections::HashMap;

use super::conv::Buffer;
use super::graph::{Act, Dense, Finish, Graph, Layer};
use super::Device;
use crate::model::Model;

pub(super) const SIZE: usize = 128;
const BLOCKS: usize = 6;
const LRELU: f32 = 0.2;
/// one style vector: the scale then the shift for 1024 channels
const STYLE: usize = 2048;

pub(super) struct Swapper {
    layers: HashMap<String, Layer>,
    /// block b's style n (1 or 2) is rows (2b + n - 1) * STYLE.. of the output
    styles: Dense,
    /// [512, 512], for the latent
    pub emap: Vec<f32>,
}

/// The swapper's output on the GPU.
pub(super) struct Encoded {
    /// f32 [128, 128, 3] in 0..1 (before clamping), in shared memory
    pub output: Buffer,
}

impl Swapper {
    pub fn load(device: &Device, model: &Model) -> Result<Swapper, String> {
        let mut layers = HashMap::new();
        let mut convs: Vec<(String, usize, usize, bool)> =
            vec![("enc.0".into(), 7, 1, true), ("enc.1".into(), 3, 1, false), ("enc.2".into(), 3, 2, false), ("enc.3".into(), 3, 2, false)];
        for b in 0..BLOCKS {
            convs.push((format!("styles.{b}.conv1"), 3, 1, true));
            convs.push((format!("styles.{b}.conv2"), 3, 1, true));
        }
        convs.extend([("dec.0".into(), 3, 1, false), ("dec.1".into(), 3, 1, false), ("dec.2".into(), 3, 1, false), ("out".into(), 7, 1, true)]);
        for (name, k, stride, reflect) in convs {
            let layer = Layer::load(device, model, &name, k, stride, reflect)?;
            layers.insert(name, layer);
        }
        let (mut weight, mut bias) = (Vec::new(), Vec::new());
        for b in 0..BLOCKS {
            for n in 1..=2 {
                let name = format!("styles.{b}.style{n}");
                let (w, bs) = (model.weight(&name)?.values, model.tensor(&format!("{name}.bias"))?.values);
                if bs.len() != STYLE {
                    return Err(format!("{name}: {} outputs, not {STYLE}", bs.len()));
                }
                weight.extend(w);
                bias.extend(bs);
            }
        }
        Ok(Swapper { layers, styles: Dense::new(device, &weight, &bias)?, emap: model.tensor("emap")?.values })
    }

    /// The style vectors for a latent (f32 [512] in `latent`).
    pub fn styles(&self, g: &mut Graph, latent: &Buffer) -> Result<Buffer, String> {
        g.dense(&self.styles, (latent, 0), false)
    }

    /// The swapper on `input` ([128, 128, 3] in 0..1) with the style vectors from `styles`, its
    /// output written at a byte offset of `dest` when given (Detail's sub-crops side by side).
    pub fn encode_to(&self, g: &mut Graph, input: &Act, styles: &Buffer, dest: Option<(&Buffer, usize)>) -> Result<Encoded, String> {
        let layer = |name: &str| self.layers.get(name).ok_or(format!("no layer {name}"));
        let style = |b: usize, n: usize| (styles, (2 * b + n) * STYLE * 4);
        let x = g.conv(layer("enc.0")?, input, LRELU)?;
        let x = g.conv(layer("enc.1")?, &x, LRELU)?;
        let x = g.conv(layer("enc.2")?, &x, LRELU)?;
        let mut x = g.conv(layer("enc.3")?, &x, LRELU)?;
        for b in 0..BLOCKS {
            let h = g.conv(layer(&format!("styles.{b}.conv1"))?, &x, 1.0)?;
            let h = g.adain(&h, style(b, 0), true, None)?;
            let h = g.conv(layer(&format!("styles.{b}.conv2"))?, &h, 1.0)?;
            x = g.adain(&h, style(b, 1), false, Some(&x))?;
        }
        let x = g.upsample(&x)?;
        let x = g.conv(layer("dec.0")?, &x, LRELU)?;
        let x = g.upsample(&x)?;
        let x = g.conv(layer("dec.1")?, &x, LRELU)?;
        let x = g.conv(layer("dec.2")?, &x, LRELU)?;
        let x = g.conv(layer("out")?, &x, 1.0)?;
        let output = match dest {
            Some((buf, offset)) => {
                g.finish_into(&x, Finish::Tanh01, (buf, offset))?;
                buf.clone()
            }
            None => g.finish(&x)?,
        };
        Ok(Encoded { output })
    }
}
