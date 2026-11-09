//! Network layers as read from a model file, f32, ready for the GPU.

use crate::model::Model;

/// A conv layer: weights [cout, k, k, cin] (as stored), padding `pad` before and after on each
/// axis, zeros or reflected.
#[derive(Clone)]
pub struct ConvLayer {
    pub k: usize,
    pub stride: usize,
    pub pad: [usize; 2],
    pub reflect: bool,
    pub cin: usize,
    pub cout: usize,
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
}

impl ConvLayer {
    /// `<name>.weight` ([cout, k*k*cin]) and `<name>.bias` (zeros when there is none).
    pub fn load(model: &Model, name: &str, k: usize, stride: usize, pad: [usize; 2], reflect: bool) -> Result<ConvLayer, String> {
        let w = model.weight(name)?;
        let cout = w.shape[0];
        let cin = w.values.len() / (cout * k * k);
        if cin * cout * k * k != w.values.len() {
            return Err(format!("{name}: weight {:?} isn't {k}x{k} with {cout} outputs", w.shape));
        }
        let bias = optional_bias(model, name, cout)?;
        Ok(ConvLayer { k, stride, pad, reflect, cin, cout, w: w.values, bias })
    }

    /// A stride-2 transposed 3x3 conv (stored [cout, 3, 3, cin], the transposed conv's weight
    /// [cin, cout, ky, kx] permuted) as a 2x2 conv to 4 * cout channels, polyphase: its pixel
    /// (a, b), channels (py * 2 + px) * cout.., is the transposed conv's pixel (2a + py, 2b + px).
    /// That output row reads input rows a - 1 + ty with kernel row py + 2 - 2 ty (taps past the
    /// kernel are zero), so the conv pads 1 before. `pad_after`: 1 for every row the transposed
    /// conv makes (2h + 1, as 2h + 2), 0 for the first 2h.
    pub fn polyphase(model: &Model, name: &str, pad_after: usize) -> Result<ConvLayer, String> {
        let t = ConvLayer::load(model, name, 3, 1, [1, 1], false)?;
        let (cin, cout) = (t.cin, t.cout);
        let mut w = vec![0f32; 4 * cout * 4 * cin];
        for q in 0..4 {
            let (py, px) = (q >> 1, q & 1);
            for ty in 0..2 {
                for tx in 0..2 {
                    let (ky, kx) = (py + 2 - 2 * ty, px + 2 - 2 * tx);
                    if ky > 2 || kx > 2 {
                        continue;
                    }
                    for co in 0..cout {
                        let src = &t.w[((co * 3 + ky) * 3 + kx) * cin..][..cin];
                        w[(((q * cout + co) * 2 + ty) * 2 + tx) * cin..][..cin].copy_from_slice(src);
                    }
                }
            }
        }
        let bias = (0..4).flat_map(|_| t.bias.iter().copied()).collect();
        Ok(ConvLayer { k: 2, stride: 1, pad: [1, pad_after], reflect: false, cin, cout: 4 * cout, w, bias })
    }
}

fn optional_bias(model: &Model, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let name = format!("{name}.bias");
    if !model.has(&name) {
        return Ok(vec![0.0; n]);
    }
    let b = model.tensor(&name)?.values;
    if b.len() != n {
        return Err(format!("{name}: {} values for {n} outputs", b.len()));
    }
    Ok(b)
}

/// A fully connected layer: weights [rows, cols], bias [rows] (zeros when there is none).
#[derive(Clone)]
pub struct Dense {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Dense {
    pub fn load(model: &Model, name: &str) -> Result<Dense, String> {
        let w = model.weight(name)?;
        let rows = w.shape[0];
        let cols = w.values.len() / rows;
        let bias = optional_bias(model, name, rows)?;
        Ok(Dense { rows, cols, w: w.values, bias })
    }

}

// ------------------------------------------------------------------------------ the enhancer

// ------------------------------------------------------------------------------ the occlusion mask
