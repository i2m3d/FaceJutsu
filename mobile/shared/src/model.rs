//! Reads model files: safetensors, with weights stored as f32, bf16, f16, or int8 in ComfyUI's
//! `comfy_quant` layout (per-row scale, optionally of a Hadamard-rotated weight, "convrot").

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

pub struct Model {
    data: Vec<u8>,
    tensors: HashMap<String, Entry>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Deserialize)]
struct QuantMeta {
    format: String,
    #[serde(default)]
    convrot: bool,
    #[serde(default)]
    convrot_groupsize: Option<usize>,
}

/// A weight as f32, with its shape.
pub struct Weight {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl Model {
    pub fn open(path: &Path) -> Result<Model, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Model::from_bytes(data)
    }

    pub fn from_bytes(data: Vec<u8>) -> Result<Model, String> {
        let n = u64::from_le_bytes(data.get(..8).ok_or("not a safetensors file")?.try_into().unwrap()) as usize;
        let header = data.get(8..8 + n).ok_or("safetensors header cut short")?;
        let mut raw: HashMap<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| format!("safetensors header: {e}"))?;
        raw.remove("__metadata__");
        let base = 8 + n;
        let mut tensors = HashMap::new();
        for (name, v) in raw {
            let mut e: Entry = serde_json::from_value(v).map_err(|e| format!("{name}: {e}"))?;
            e.data_offsets = [e.data_offsets[0] + base, e.data_offsets[1] + base];
            if e.data_offsets[1] > data.len() {
                return Err(format!("{name}: data past the end of the file"));
            }
            tensors.insert(name, e);
        }
        Ok(Model { data, tensors })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn raw(&self, name: &str) -> Result<(&Entry, &[u8]), String> {
        let e = self.tensors.get(name).ok_or(format!("no tensor {name}"))?;
        Ok((e, &self.data[e.data_offsets[0]..e.data_offsets[1]]))
    }

    /// A plain tensor as f32.
    pub fn tensor(&self, name: &str) -> Result<Weight, String> {
        let (e, bytes) = self.raw(name)?;
        let values = match e.dtype.as_str() {
            "F32" => bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            "F16" => bytes.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            "BF16" => bytes.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            "I8" => bytes.iter().map(|b| *b as i8 as f32).collect(),
            d => return Err(format!("{name}: dtype {d}")),
        };
        Ok(Weight { shape: e.shape.clone(), values })
    }

    /// A layer's weight (`<layer>.weight`) as f32, dequantized when stored quantized.
    pub fn weight(&self, layer: &str) -> Result<Weight, String> {
        let name = format!("{layer}.weight");
        let meta_name = format!("{layer}.comfy_quant");
        if !self.has(&meta_name) {
            return self.tensor(&name);
        }
        let (_, meta_bytes) = self.raw(&meta_name)?;
        let meta: QuantMeta = serde_json::from_slice(meta_bytes).map_err(|e| format!("{meta_name}: {e}"))?;
        if meta.format != "int8_tensorwise" {
            return Err(format!("{layer}: quantization {} isn't supported yet", meta.format));
        }
        let mut w = self.tensor(&name)?;
        let scale = self.tensor(&format!("{layer}.weight_scale"))?.values;
        let (rows, cols) = (w.shape[0], w.values.len() / w.shape[0]);
        if scale.len() != rows {
            return Err(format!("{layer}: {} scales for {rows} rows", scale.len()));
        }
        for (row, s) in w.values.chunks_exact_mut(cols).zip(&scale) {
            row.iter_mut().for_each(|v| *v *= s);
        }
        if meta.convrot {
            let group = meta.convrot_groupsize.unwrap_or(256);
            if group != 256 || cols % group != 0 {
                return Err(format!("{layer}: convrot group {group} over {cols} columns"));
            }
            w.values.chunks_exact_mut(group).for_each(hadamard_256);
        }
        Ok(w)
    }
}

/// Multiplies 256 values by the normalized regular Hadamard matrix H4 x H4 x H4 x H4 (symmetric
/// and orthogonal, so it undoes itself): H4 along each base-4 digit of the index.
fn hadamard_256(x: &mut [f32]) {
    for stride in [1, 4, 16, 64] {
        for block in (0..256).step_by(stride * 4) {
            for i in block..block + stride {
                let (a, b, c, d) = (x[i], x[i + stride], x[i + 2 * stride], x[i + 3 * stride]);
                x[i] = a + b + c - d;
                x[i + stride] = a + b - c + d;
                x[i + 2 * stride] = a - b + c + d;
                x[i + 3 * stride] = -a + b + c + d;
            }
        }
    }
    x.iter_mut().for_each(|v| *v /= 16.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_undoes_itself() {
        let orig: Vec<f32> = (0..256).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let mut x = orig.clone();
        hadamard_256(&mut x);
        assert!(x.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-3));
        hadamard_256(&mut x);
        assert!(x.iter().zip(&orig).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn hadamard_keeps_length() {
        let mut x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let before: f32 = x.iter().map(|v| v * v).sum();
        hadamard_256(&mut x);
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-3 * before);
    }
}
