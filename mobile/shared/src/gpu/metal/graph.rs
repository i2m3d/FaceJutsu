//! Building blocks for running networks and image steps on the GPU: conv layers loaded from a
//! model, activations (f16 NHWC, batch 1), 8-bit images, and a graph that encodes dispatches, in
//! order, into one command buffer. Convs run on the tensor-op kernel; the rest on the small kernels
//! in ops.metal and image.metal.
//!
//! A graph can also run in profile mode: each step then gets its own command buffer, so its GPU
//! time is measured on its own (the total is a little higher than when encoded together).

use std::ffi::c_void;
use std::ptr::NonNull;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

use super::conv::{buffer_with, Buffer, Kernels, Queue, TensorParams, Variant};
use super::{library, pipeline, Device, Pipeline};
use crate::model::Model;
use crate::photo::Affine;

const SOURCE: &str = concat!(include_str!("ops.metal"), include_str!("image.metal"));
const RESTORE: &str = include_str!("restore.metal");

type Commands = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// A conv layer's weights on the GPU, and its padding before and after on each axis.
pub(super) struct Layer {
    k: usize,
    stride: usize,
    pad: [usize; 2],
    reflect: bool,
    cp: usize,
    pub cout: usize,
    wt: Buffer,
    bias: Buffer,
}

impl Layer {
    /// Loads `<name>.weight` ([Cout, k, k, Cin], dequantized) and `<name>.bias`, padded by
    /// (k - 1) / 2.
    pub fn load(device: &Device, model: &Model, name: &str, k: usize, stride: usize, reflect: bool) -> Result<Layer, String> {
        let p = (k - 1) / 2;
        Layer::from_cpu(device, &crate::cpu::ConvLayer::load(model, name, k, stride, [p, p], reflect)?)
    }

    /// The CPU's layer on the GPU.
    pub fn from_cpu(device: &Device, l: &crate::cpu::ConvLayer) -> Result<Layer, String> {
        let (k, cin, cout) = (l.k, l.cin, l.cout);
        let cp = cin.next_multiple_of(8);
        let mut wt = vec![0u16; cout * k * k * cp];
        for (tap, src) in l.w.chunks_exact(cin).enumerate() {
            for (d, v) in wt[tap * cp..][..cin].iter_mut().zip(src) {
                *d = f16::from_f32(*v).to_bits();
            }
        }
        Ok(Layer { k, stride: l.stride, pad: l.pad, reflect: l.reflect, cp, cout, wt: buffer_with(device, &wt)?, bias: buffer_with(device, &l.bias)? })
    }
}

/// A fully connected layer on the GPU: f16 weights [rows, cols], f32 bias.
pub(super) struct Dense {
    wt: Buffer,
    bias: Buffer,
    pub rows: usize,
    pub cols: usize,
}

impl Dense {
    pub fn new(device: &Device, weight: &[f32], bias: &[f32]) -> Result<Dense, String> {
        let rows = bias.len();
        if rows == 0 || weight.len() % rows != 0 {
            return Err(format!("dense: {} weights for {rows} outputs", weight.len()));
        }
        let halves: Vec<u16> = weight.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        Ok(Dense { wt: buffer_with(device, &halves)?, bias: buffer_with(device, bias)?, rows, cols: weight.len() / rows })
    }

    pub fn from_cpu(device: &Device, d: &crate::cpu::Dense) -> Result<Dense, String> {
        Dense::new(device, &d.w, &d.bias)
    }
}

/// An activation: [h, w, c] f16, rows stored `sw` wide.
#[derive(Clone)]
pub(super) struct Act {
    pub buf: Buffer,
    pub h: usize,
    pub w: usize,
    pub sw: usize,
    pub c: usize,
}

impl Act {
    /// Uploads compact [h, w, c] values.
    pub fn upload(device: &Device, h: usize, w: usize, c: usize, values: &[f32]) -> Result<Act, String> {
        let halves: Vec<u16> = values.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
        Ok(Act { buf: buffer_with(device, &halves)?, h, w, sw: w, c })
    }

}

/// The first `n` values of a shared buffer the GPU is done with.
pub(super) fn host<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    assert!(n * size_of::<T>() <= buf.length());
    // SAFETY: shared memory of at least n values of T, which no command buffer is writing.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n) }.to_vec()
}

/// An 8-bit image on the GPU: RGB or BGRA, rows `stride` bytes apart.
pub(super) struct Img8 {
    pub buf: Buffer,
    pub h: usize,
    pub w: usize,
    pub stride: usize,
    pub bgra: bool,
}

impl Img8 {

    /// An image in shared memory, to fill from the CPU or to read there once a run is done.
    pub fn shared(device: &Device, h: usize, w: usize, stride: usize, bgra: bool) -> Result<Img8, String> {
        if stride < w * if bgra { 4 } else { 3 } {
            return Err(format!("rows of {stride} bytes for a {w} px wide image"));
        }
        let buf = device.newBufferWithLength_options((h * stride).max(4), MTLResourceOptions::StorageModeShared).ok_or("out of GPU memory")?;
        Ok(Img8 { buf, h, w, stride, bgra })
    }

    /// The image's bytes in shared memory (h * stride of them).
    pub fn bytes(&self) -> *mut u8 {
        self.buf.contents().as_ptr() as *mut u8
    }
}

#[repr(C)]
struct CopyParams {
    rows: u32,
    row_bytes: u32,
    src_stride: u32,
    dst_stride: u32,
}

/// The small kernels, compiled once.
pub(super) struct Ops {
    pad: Pipeline,
    stats: Pipeline,
    adain: Pipeline,
    upsample: Pipeline,
    finish: Pipeline,
    affine: Pipeline,
    pool: Pipeline,
    up_add: Pipeline,
    resample: Pipeline,
    matvec_f32: Pipeline,
    matvec_f16: Pipeline,
    paste: Pipeline,
    copy: Pipeline,
    copy32: Pipeline,
    to_host: Pipeline,
    /// the enhancer's and the occlusion mask's, or why they didn't compile
    restore: Result<RestoreOps, String>,
}

/// The enhancer's and the occlusion mask's kernels (restore.metal), compiled on their own.
struct RestoreOps {
    fir_down: Pipeline,
    styled_act: Pipeline,
    to_rgb: Pipeline,
    pixel_norm: Pipeline,
    square: Pipeline,
    frn_stats: Pipeline,
    frn_out: Pipeline,
    blur_down: Pipeline,
    to_half: Pipeline,
    mask_morph: Pipeline,
    mask_blur: Pipeline,
    unocclude: Pipeline,
    band_down: Pipeline,
    enhance_out: Pipeline,
}

impl RestoreOps {
    fn new(device: &Device) -> Result<RestoreOps, String> {
        let lib = library(device, RESTORE, None)?;
        let c = |name: &str| pipeline(device, &lib, name);
        let ops = RestoreOps {
            fir_down: c("fir_down")?,
            styled_act: c("styled_act")?,
            to_rgb: c("to_rgb")?,
            pixel_norm: c("pixel_norm")?,
            square: c("square_values")?,
            frn_stats: c("frn_stats")?,
            frn_out: c("frn_out")?,
            blur_down: c("blur_down")?,
            to_half: c("to_half")?,
            mask_morph: c("mask_morph")?,
            mask_blur: c("mask_blur")?,
            unocclude: c("unocclude")?,
            band_down: c("band_down")?,
            enhance_out: c("enhance_out")?,
        };
        if ops.frn_stats.maxTotalThreadsPerThreadgroup() < 1024 {
            return Err(format!("frn_stats needs 1024 threads per threadgroup, gets {}", ops.frn_stats.maxTotalThreadsPerThreadgroup()));
        }
        Ok(ops)
    }
}

impl Ops {
    fn restore(&self) -> Result<&RestoreOps, String> {
        self.restore.as_ref().map_err(|e| format!("the enhancer's and the mask's kernels: {e}"))
    }

    pub fn new(device: &Device) -> Result<Ops, String> {
        let lib = library(device, SOURCE, None)?;
        let c = |name: &str| pipeline(device, &lib, name);
        let ops = Ops {
            restore: RestoreOps::new(device),
            pad: c("pad")?,
            stats: c("instance_stats")?,
            adain: c("adain")?,
            upsample: c("upsample2")?,
            finish: c("finish")?,
            affine: c("affine_act")?,
            pool: c("pool2")?,
            up_add: c("up_add")?,
            resample: c("resample_rgb8")?,
            matvec_f32: c("matvec_f32")?,
            matvec_f16: c("matvec_f16")?,
            paste: c("paste")?,
            copy: c("copy_rows")?,
            copy32: c("copy_rows32")?,
            to_host: c("act_to_f32")?,
        };
        if ops.stats.maxTotalThreadsPerThreadgroup() < 1024 {
            return Err(format!("instance_stats needs 1024 threads per threadgroup, gets {}", ops.stats.maxTotalThreadsPerThreadgroup()));
        }
        Ok(ops)
    }
}

#[repr(C)]
struct PadParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    cp: u32,
    lo: u32,
    hi: u32,
    reflect: u32,
    scaled: u32,
}

#[repr(C)]
struct FirParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
}

#[repr(C)]
struct StyledParams {
    h: u32,
    w: u32,
    c: u32,
    ysw: u32,
    fsw: u32,
    up: u32,
    noise: f32,
}

#[repr(C)]
struct RgbParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    skip: u32,
}

#[repr(C)]
struct FrnParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    groups: u32,
    eps: f32,
}

#[repr(C)]
struct FrnOutParams {
    oh: u32,
    ow: u32,
    c: u32,
    cs: u32,
    xsw: u32,
    shuffle: u32,
}

#[repr(C)]
struct BlurDownParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    taps: u32,
    k: [f32; 4],
}

#[repr(C)]
struct MaskParams {
    n: u32,
    r: u32,
    erode: u32,
    vertical: u32,
    finish: u32,
}

#[repr(C)]
struct UnoccludeParams {
    m: [f32; 6],
    x1: u32,
    y1: u32,
    bw: u32,
    bh: u32,
    stride: u32,
    bgra: u32,
    ostride: u32,
    obgra: u32,
    n: u32,
}

#[repr(C)]
struct BandParams {
    n: u32,
    lo: u32,
    k: u32,
    s: f32,
}

#[repr(C)]
struct NormParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    relu: u32,
    residual: u32,
}

#[repr(C)]
struct UpParams {
    h: u32,
    w: u32,
    c: u32,
}

#[repr(C)]
struct FinishParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    mode: u32,
}

/// How `finish` maps a network's output.
#[derive(Clone, Copy)]
pub(super) enum Finish {
    /// (tanh + 1) / 2
    Tanh01 = 0,
    Sigmoid = 1,
}

#[repr(C)]
struct AffineParams {
    h: u32,
    w: u32,
    c: u32,
    sw_a: u32,
    sw_b: u32,
    scale: u32,
    add: u32,
    act: u32,
}

#[repr(C)]
struct PoolParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    avg: u32,
}

#[repr(C)]
struct UpAddParams {
    h: u32,
    w: u32,
    sw: u32,
    c: u32,
    bh: u32,
    bw: u32,
    bsw: u32,
}

#[repr(C)]
struct ResampleParams {
    m: [f32; 6],
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    n: u32,
    valid_w: u32,
    valid_h: u32,
    mul: f32,
    add: f32,
    stride: u32,
    bgra: u32,
    k: u32,
    si: u32,
    sj: u32,
}

#[repr(C)]
struct MatvecParams {
    rows: u32,
    cols: u32,
    slope: f32,
}

#[repr(C)]
struct PasteParams {
    m: [f32; 6],
    x1: u32,
    y1: u32,
    bw: u32,
    bh: u32,
    stride: u32,
    bgra: u32,
    s: u32,
    k: u32,
    alpha: f32,
    add: u32,
}

/// What follows an affine op's sum.
pub(super) enum Activation<'b> {
    None,
    Relu,
    Prelu(&'b Buffer),
}

/// How an image is mapped into a network input: out = value * mul + add, black outside `valid`.
pub(super) struct Sampling {
    pub map: Affine,
    pub size: (usize, usize),
    pub valid: (usize, usize),
    pub mul: f32,
    pub add: f32,
}

/// The buffers of a graph that runs again and again, such as one per video frame: a run's i-th
/// buffer is the last run's i-th when it's the same size, so a run allocates nothing once the
/// sizes settle. What a run returns is only good until the next run with the same arena.
#[derive(Default)]
pub(super) struct Arena {
    bufs: Vec<(usize, bool, Buffer)>,
}

/// Encodes dispatches, in order, into a command buffer (one per step in profile mode).
pub(super) struct Graph<'g> {
    pub kernels: &'g mut Kernels,
    pub ops: &'g Ops,
    queue: Queue,
    commands: Commands,
    encoder: Encoder,
    /// what the command buffer uses, kept until it's done: in the arena, or here without one
    keep: Vec<Buffer>,
    arena: Option<&'g mut Arena>,
    next: usize,
    /// per step, (kind, GPU ms), when profiling
    profile: Option<Vec<(&'static str, f64)>>,
}

impl Graph<'_> {
    fn buffer(&mut self, bytes: usize, shared: bool) -> Result<Buffer, String> {
        let i = self.next;
        self.next += 1;
        if let Some((n, s, b)) = self.arena.as_ref().and_then(|a| a.bufs.get(i)) {
            if (*n, *s) == (bytes, shared) {
                return Ok(b.clone());
            }
        }
        let mode = if shared { MTLResourceOptions::StorageModeShared } else { MTLResourceOptions::StorageModePrivate };
        let b = self.kernels.device.newBufferWithLength_options(bytes.max(4), mode).ok_or("out of GPU memory")?;
        match self.arena.as_deref_mut() {
            // every request takes the next index, so i is at most the length
            Some(a) if i < a.bufs.len() => a.bufs[i] = (bytes, shared, b.clone()),
            Some(a) => a.bufs.push((bytes, shared, b.clone())),
            None => self.keep.push(b.clone()),
        }
        Ok(b)
    }

    /// Marks the end of a step: in profile mode, runs it and times it.
    fn done(&mut self, kind: &'static str) -> Result<(), String> {
        if self.profile.is_none() {
            return Ok(());
        }
        self.encoder.endEncoding();
        self.commands.commit();
        self.commands.waitUntilCompleted();
        if let Some(e) = self.commands.error() {
            return Err(e.localizedDescription().to_string());
        }
        let ms = (self.commands.GPUEndTime() - self.commands.GPUStartTime()) * 1e3;
        self.profile.as_mut().unwrap().push((kind, ms));
        self.commands = self.queue.commandBuffer().ok_or("no command buffer")?;
        self.encoder = self.commands.computeCommandEncoder().ok_or("no encoder")?;
        Ok(())
    }

    /// Sets buffers (with offsets) 0.., then the params after them, and dispatches `groups`.
    fn dispatch<P>(&self, pipeline: &Pipeline, buffers: &[(&Buffer, usize)], params: &P, groups: MTLSize, threads: MTLSize) {
        self.encoder.setComputePipelineState(pipeline);
        // SAFETY: the buffers are kept alive until the command buffer completes, and are sized
        // for what the kernel reads and writes with these params.
        unsafe {
            for (i, (b, offset)) in buffers.iter().enumerate() {
                self.encoder.setBuffer_offset_atIndex(Some(b), *offset, i);
            }
            self.encoder.setBytes_length_atIndex(NonNull::from(params).cast::<c_void>(), size_of::<P>(), buffers.len());
        }
        self.encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
    }

    /// One thread per value, in threadgroups of 256.
    fn elementwise<P>(&self, pipeline: &Pipeline, buffers: &[&Buffer], params: &P, n: usize) {
        let with_offsets: Vec<(&Buffer, usize)> = buffers.iter().map(|b| (*b, 0)).collect();
        let groups = MTLSize { width: n.div_ceil(256), height: 1, depth: 1 };
        self.dispatch(pipeline, &with_offsets, params, groups, MTLSize { width: 256, height: 1, depth: 1 });
    }

    /// Pads `x` as the layer needs and runs the conv with bias and leaky ReLU (`slope`: 0 for
    /// ReLU, 1 for none).
    pub fn conv(&mut self, layer: &Layer, x: &Act, slope: f32) -> Result<Act, String> {
        self.conv_scaled(layer, x, slope, None)
    }

    /// `conv` with each input channel first multiplied by `scale` (f32 at a byte offset): a
    /// modulated conv's style.
    pub fn conv_scaled(&mut self, layer: &Layer, x: &Act, slope: f32, scale: Option<(&Buffer, usize)>) -> Result<Act, String> {
        let (k, s, cp, cout) = (layer.k, layer.stride, layer.cp, layer.cout);
        if x.c > cp || x.c.next_multiple_of(8) != cp {
            return Err(format!("a conv for {cp} input channels given {}", x.c));
        }
        let [lo, hi] = layer.pad;
        let (ph, pw) = (x.h + lo + hi, x.w + lo + hi);
        let padded = self.buffer(ph * pw * cp * 2, false)?;
        let pp = PadParams {
            h: x.h as u32,
            w: x.w as u32,
            sw: x.sw as u32,
            c: x.c as u32,
            cp: cp as u32,
            lo: lo as u32,
            hi: hi as u32,
            reflect: layer.reflect as u32,
            scaled: scale.is_some() as u32,
        };
        let sc = scale.unwrap_or((&padded, 0));
        let groups = MTLSize { width: (ph * pw * cp).div_ceil(256), height: 1, depth: 1 };
        self.dispatch(&self.ops.pad, &[(&x.buf, 0), (&padded, 0), sc], &pp, groups, MTLSize { width: 256, height: 1, depth: 1 });
        self.done("pad")?;

        let (out_h, out_w) = ((ph - k) / s + 1, (pw - k) / s + 1);
        let v = Variant::for_shape(s, out_w, cout);
        let sw = if v.wide { pw } else { out_w };
        let m = out_h * sw;
        let y = self.buffer(m * cout * 2, false)?;
        let params = TensorParams {
            in_w: pw as u32,
            out_h: out_h as u32,
            out_w: out_w as u32,
            m: m as u32,
            cin: cp as u32,
            cout: cout as u32,
            k: k as u32,
            stride: s as u32,
            slope,
        };
        let (pipeline, _) = self.kernels.pipelines(v)?;
        let tiles = if v.wide { (out_h * pw - (k - 1)).div_ceil(v.bm) } else { out_h * out_w.div_ceil(v.bm) };
        let groups = MTLSize { width: tiles, height: cout.div_ceil(v.bn), depth: 1 };
        // buffer 5 (split partial sums) is unused without a split
        self.encoder.setComputePipelineState(&pipeline);
        // SAFETY: as in dispatch.
        unsafe {
            for (i, b) in [&padded, &layer.wt, &layer.bias, &y].into_iter().enumerate() {
                self.encoder.setBuffer_offset_atIndex(Some(b), 0, i);
            }
            self.encoder.setBytes_length_atIndex(NonNull::from(&params).cast::<c_void>(), size_of::<TensorParams>(), 4);
            self.encoder.setBuffer_offset_atIndex(Some(&y), 0, 5);
        }
        self.encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, MTLSize { width: 128, height: 1, depth: 1 });
        self.done("conv")?;
        Ok(Act { buf: y, h: out_h, w: out_w, sw, c: cout })
    }

    /// Instance norm with an AdaIN style (scale then shift, c each, at `style_offset` bytes into
    /// `style`), optional ReLU and residual.
    pub fn adain(&mut self, x: &Act, style: (&Buffer, usize), relu: bool, residual: Option<&Act>) -> Result<Act, String> {
        let stats = self.buffer(x.c * 8, false)?;
        let np = NormParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32, relu: relu as u32, residual: residual.is_some() as u32 };
        let groups = MTLSize { width: x.c.div_ceil(32), height: 1, depth: 1 };
        self.dispatch(&self.ops.stats, &[(&x.buf, 0), (&stats, 0)], &np, groups, MTLSize { width: 32, height: 32, depth: 1 });
        self.done("instance stats")?;
        let out = self.buffer(x.h * x.w * x.c * 2, false)?;
        let res = residual.map(|r| r.buf.clone()).unwrap_or_else(|| x.buf.clone());
        let n = x.h * x.w * x.c;
        let groups = MTLSize { width: n.div_ceil(256), height: 1, depth: 1 };
        self.dispatch(&self.ops.adain, &[(&x.buf, 0), (&stats, 0), style, (&res, 0), (&out, 0)], &np, groups, MTLSize { width: 256, height: 1, depth: 1 });
        self.done("adain")?;
        Ok(Act { buf: out, h: x.h, w: x.w, sw: x.w, c: x.c })
    }

    /// Bilinear upsampling by 2 of a compact activation.
    pub fn upsample(&mut self, x: &Act) -> Result<Act, String> {
        if x.sw != x.w {
            return Err("upsample needs a compact input".into());
        }
        let out = self.buffer(4 * x.h * x.w * x.c * 2, false)?;
        self.elementwise(&self.ops.upsample, &[&x.buf, &out], &UpParams { h: x.h as u32, w: x.w as u32, c: x.c as u32 }, 4 * x.h * x.w * x.c);
        self.done("upsample")?;
        Ok(Act { buf: out, h: 2 * x.h, w: 2 * x.w, sw: 2 * x.w, c: x.c })
    }

    /// (tanh + 1) / 2, into an f32 buffer of compact [h, w, c].
    pub fn finish(&mut self, x: &Act) -> Result<Buffer, String> {
        let out = self.buffer(x.h * x.w * x.c * 4, true)?;
        self.finish_into(x, Finish::Tanh01, (&out, 0))?;
        Ok(out)
    }

    /// A network's output mapped as `mode` says, into f32 compact [h, w, c] at a byte offset of
    /// `out`.
    pub fn finish_into(&mut self, x: &Act, mode: Finish, out: (&Buffer, usize)) -> Result<(), String> {
        let n = x.h * x.w * x.c;
        let p = FinishParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32, mode: mode as u32 };
        let groups = MTLSize { width: n.div_ceil(256), height: 1, depth: 1 };
        self.dispatch(&self.ops.finish, &[(&x.buf, 0), out], &p, groups, MTLSize { width: 256, height: 1, depth: 1 });
        self.done("finish")
    }

    /// An f32 buffer of `n` values (the GPU's own, private).
    pub fn f32_buffer(&mut self, n: usize) -> Result<Buffer, String> {
        self.buffer(n * 4, false)
    }

    /// act(a * scale + shift + b): `scale` is (scale, shift) per channel.
    pub fn affine(&mut self, a: &Act, scale: Option<(&Buffer, &Buffer)>, b: Option<&Act>, act: Activation) -> Result<Act, String> {
        if let Some(b) = b {
            if (b.h, b.w, b.c) != (a.h, a.w, a.c) {
                return Err(format!("add: {}x{}x{} and {}x{}x{}", a.h, a.w, a.c, b.h, b.w, b.c));
            }
        }
        let out = self.buffer(a.h * a.w * a.c * 2, false)?;
        let (sc, sh) = scale.map(|(s, t)| (s.clone(), t.clone())).unwrap_or_else(|| (out.clone(), out.clone()));
        let (code, alpha) = match act {
            Activation::None => (0, out.clone()),
            Activation::Relu => (1, out.clone()),
            Activation::Prelu(al) => (2, al.clone()),
        };
        let bb = b.map(|b| b.buf.clone()).unwrap_or_else(|| a.buf.clone());
        let p = AffineParams {
            h: a.h as u32,
            w: a.w as u32,
            c: a.c as u32,
            sw_a: a.sw as u32,
            sw_b: b.map_or(a.sw, |b| b.sw) as u32,
            scale: scale.is_some() as u32,
            add: b.is_some() as u32,
            act: code,
        };
        self.elementwise(&self.ops.affine, &[&a.buf, &bb, &sc, &sh, &alpha, &out], &p, a.h * a.w * a.c);
        self.done("affine")?;
        Ok(Act { buf: out, h: a.h, w: a.w, sw: a.w, c: a.c })
    }

    /// 2x2 pooling, stride 2, output size rounded up: average or max.
    pub fn pool(&mut self, x: &Act, avg: bool) -> Result<Act, String> {
        let (oh, ow) = (x.h.div_ceil(2), x.w.div_ceil(2));
        let out = self.buffer(oh * ow * x.c * 2, false)?;
        let p = PoolParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32, avg: avg as u32 };
        self.elementwise(&self.ops.pool, &[&x.buf, &out], &p, oh * ow * x.c);
        self.done("pool")?;
        Ok(Act { buf: out, h: oh, w: ow, sw: ow, c: x.c })
    }

    /// a + b upsampled to a's size by nearest neighbour.
    pub fn up_add(&mut self, a: &Act, b: &Act) -> Result<Act, String> {
        let out = self.buffer(a.h * a.w * a.c * 2, false)?;
        let p = UpAddParams { h: a.h as u32, w: a.w as u32, sw: a.sw as u32, c: a.c as u32, bh: b.h as u32, bw: b.w as u32, bsw: b.sw as u32 };
        self.elementwise(&self.ops.up_add, &[&a.buf, &b.buf, &out], &p, a.h * a.w * a.c);
        self.done("upsample add")?;
        Ok(Act { buf: out, h: a.h, w: a.w, sw: a.w, c: a.c })
    }

    /// Resamples an 8-bit image into a network input (f16, 3 channels).
    pub fn resample(&mut self, img: &Img8, s: &Sampling) -> Result<Act, String> {
        self.resample_sub(img, s, 1, 0, 0)
    }

    /// Detail: sub-crop (si, sj) of the crop `s` describes (k times the sub-crop's size; `s.size`
    /// is the sub-crop's): its pixel (x, y) is the crop's (x k + si, y k + sj).
    pub fn resample_sub(&mut self, img: &Img8, s: &Sampling, k: usize, si: usize, sj: usize) -> Result<Act, String> {
        let (oh, ow) = s.size;
        let m = s.map;
        let scale = (m[0] * m[4] - m[1] * m[3]).abs().sqrt();
        let n = ((scale - 1e-6).ceil() as usize).clamp(1, 4);
        let out = self.buffer(oh * ow * 3 * 2, false)?;
        let p = ResampleParams {
            m: m.map(|v| v as f32),
            h: img.h as u32,
            w: img.w as u32,
            oh: oh as u32,
            ow: ow as u32,
            n: n as u32,
            // (in the crop's pixels)
            valid_w: (s.valid.0 * k) as u32,
            valid_h: (s.valid.1 * k) as u32,
            mul: s.mul,
            add: s.add,
            stride: img.stride as u32,
            bgra: img.bgra as u32,
            k: k as u32,
            si: si as u32,
            sj: sj as u32,
        };
        self.elementwise(&self.ops.resample, &[&img.buf, &out], &p, oh * ow);
        self.done("resample")?;
        Ok(Act { buf: out, h: oh, w: ow, sw: ow, c: 3 })
    }

    /// layer(x): an f32 vector (`x` at `offset` bytes) or a compact activation flattened.
    pub fn dense(&mut self, layer: &Dense, x: (&Buffer, usize), half_input: bool) -> Result<Buffer, String> {
        self.dense_act(layer, x, half_input, 1.0)
    }

    /// `dense` then a leaky ReLU with `slope`.
    pub fn dense_act(&mut self, layer: &Dense, x: (&Buffer, usize), half_input: bool, slope: f32) -> Result<Buffer, String> {
        let out = self.buffer(layer.rows * 4, true)?;
        let p = MatvecParams { rows: layer.rows as u32, cols: layer.cols as u32, slope };
        let pipeline = if half_input { &self.ops.matvec_f16 } else { &self.ops.matvec_f32 };
        let groups = MTLSize { width: layer.rows.div_ceil(8), height: 1, depth: 1 };
        self.dispatch(pipeline, &[(&layer.wt, 0), x, (&layer.bias, 0), (&out, 0)], &p, groups, MTLSize { width: 256, height: 1, depth: 1 });
        self.done("dense")?;
        Ok(out)
    }

    /// A compact f32 copy of an activation in shared memory, for the CPU to read once the run is
    /// done (see `floats`).
    pub fn to_host(&mut self, x: &Act) -> Result<Buffer, String> {
        let out = self.buffer(x.h * x.w * x.c * 4, true)?;
        self.elementwise(&self.ops.to_host, &[&x.buf, &out], &FinishParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32, mode: 0 }, x.h * x.w * x.c);
        self.done("to host")?;
        Ok(out)
    }

    /// Copies an 8-bit image into another of its size and layout.
    pub fn copy(&mut self, src: &Img8, dst: &Img8) -> Result<(), String> {
        if (src.w, src.h, src.bgra) != (dst.w, dst.h, dst.bgra) {
            return Err(format!("copying a {}x{} image into a {}x{} one of another layout", src.w, src.h, dst.w, dst.h));
        }
        let row_bytes = src.w * if src.bgra { 4 } else { 3 };
        let p = CopyParams { rows: src.h as u32, row_bytes: row_bytes as u32, src_stride: src.stride as u32, dst_stride: dst.stride as u32 };
        if (row_bytes | src.stride | dst.stride) % 4 == 0 {
            self.elementwise(&self.ops.copy32, &[&src.buf, &dst.buf], &p, src.h * row_bytes / 4);
        } else {
            self.elementwise(&self.ops.copy, &[&src.buf, &dst.buf], &p, src.h * row_bytes);
        }
        self.done("copy")?;
        Ok(())
    }

    /// Blends a swapped crop (f32 [s, s, 3], 0..1, held as k x k sub-crops for Detail) into an
    /// 8-bit frame, in place, through the inverse of `m` (frame -> crop), with the edge mask from
    /// `profile` (s values) times `alpha`; `add`: the crop is a change to the frame (0..1 units),
    /// added.
    #[allow(clippy::too_many_arguments)]
    pub fn paste_with(&mut self, frame: &Img8, crop: &Buffer, s: usize, k: usize, profile: &Buffer, m: &Affine, alpha: f32, add: bool) -> Result<(), String> {
        let [x1, y1, x2, y2] = crate::photo::paste_box(m, s, frame.w, frame.h);
        if x2 <= x1 || y2 <= y1 {
            return Ok(());
        }
        let p = PasteParams {
            m: m.map(|v| v as f32),
            x1: x1 as u32,
            y1: y1 as u32,
            bw: (x2 - x1) as u32,
            bh: (y2 - y1) as u32,
            stride: frame.stride as u32,
            bgra: frame.bgra as u32,
            s: s as u32,
            k: k as u32,
            alpha,
            add: add as u32,
        };
        self.elementwise(&self.ops.paste, &[&frame.buf, crop, profile], &p, (x2 - x1) * (y2 - y1));
        self.done("paste")?;
        Ok(())
    }

    // ------------------------------------------------------------------ the enhancer

    /// The enhancer encoder's blur before a stride-2 conv: [h, w, c] -> [h + 1, w + 1, c].
    pub fn fir_down(&mut self, x: &Act) -> Result<Act, String> {
        let (oh, ow) = (x.h + 1, x.w + 1);
        let out = self.buffer(oh * ow * x.c * 2, false)?;
        let p = FirParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32 };
        self.elementwise(&self.ops.restore()?.fir_down, &[&x.buf, &out], &p, oh * ow * x.c);
        self.done("blur")?;
        Ok(Act { buf: out, h: oh, w: ow, sw: ow, c: x.c })
    }

    /// A modulated conv's output `y` through the enhancer's activation, joined with `feat`
    /// times `noise`: [h, w, 2c] at feat's size. `d`: the demodulation sums (f32 [c]); `act`:
    /// the bias (f32 [2c]); `up`: y is the polyphase transposed conv.
    pub fn styled_act(&mut self, y: &Act, d: &Buffer, act: &Buffer, noise: f32, feat: &Act, up: bool) -> Result<Act, String> {
        let c = feat.c;
        if y.c != if up { 4 * c } else { c } {
            return Err(format!("styled activation: {} conv channels for a {c}-channel map", y.c));
        }
        let out = self.buffer(feat.h * feat.w * 2 * c * 2, false)?;
        let p = StyledParams { h: feat.h as u32, w: feat.w as u32, c: c as u32, ysw: y.sw as u32, fsw: feat.sw as u32, up: up as u32, noise };
        self.elementwise(&self.ops.restore()?.styled_act, &[&y.buf, d, act, &feat.buf, &out], &p, feat.h * feat.w * 2 * c);
        self.done("styled activation")?;
        Ok(Act { buf: out, h: feat.h, w: feat.w, sw: feat.w, c: 2 * c })
    }

    /// The modulated 1x1 conv to RGB (style f32 at a byte offset of `style`; `wt` f32 [3, c],
    /// `bias` [3]), plus `skip` (f32 [h/2, w/2, 3]) upsampled: f32 compact [h, w, 3].
    pub fn to_rgb(&mut self, x: &Act, style: (&Buffer, usize), wt: &Buffer, bias: &Buffer, skip: Option<&Buffer>) -> Result<Buffer, String> {
        let out = self.buffer(x.h * x.w * 3 * 4, false)?;
        let p = RgbParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32, skip: skip.is_some() as u32 };
        let sk = skip.unwrap_or(&out);
        let n = x.h * x.w;
        let groups = MTLSize { width: n.div_ceil(256), height: 1, depth: 1 };
        self.dispatch(&self.ops.restore()?.to_rgb, &[(&x.buf, 0), style, (wt, 0), (bias, 0), (sk, 0), (&out, 0)], &p, groups, MTLSize { width: 256, height: 1, depth: 1 });
        self.done("to rgb")?;
        Ok(out)
    }

    /// Pixel norm of an f32 vector of `n` values, in place.
    pub fn pixel_norm(&mut self, x: &Buffer, n: usize) -> Result<(), String> {
        let one = MTLSize { width: 1, height: 1, depth: 1 };
        self.dispatch(&self.ops.restore()?.pixel_norm, &[(x, 0)], &(n as u32), one, MTLSize { width: 32, height: 1, depth: 1 });
        self.done("pixel norm")
    }

    /// x * x for an f32 vector of `n` values.
    pub fn square(&mut self, x: &Buffer, n: usize) -> Result<Buffer, String> {
        let out = self.buffer(n * 4, false)?;
        self.elementwise(&self.ops.restore()?.square, &[x, &out], &(n as u32), n);
        self.done("square")?;
        Ok(out)
    }

    /// The enhancer's detail band: the part of c - g (its input, and its output f32 [n, n, 3])
    /// below `lo` px, f32 [lo, lo, 3].
    pub fn band_down(&mut self, c: &Act, g: &Buffer, lo: usize) -> Result<Buffer, String> {
        let n = c.w;
        let out = self.buffer(lo * lo * 3 * 4, false)?;
        let s = n as f32 / lo as f32;
        let p = BandParams { n: n as u32, lo: lo as u32, k: ((s - 1e-6).ceil() as u32).clamp(1, 4), s };
        self.elementwise(&self.ops.restore()?.band_down, &[&c.buf, g, &out], &p, lo * lo);
        self.done("band")?;
        Ok(out)
    }

    /// The enhancer's output for the paste (f32 [n, n, 3], 0..1 units): with `band` (from
    /// band_down, and `c` its input) its change to the input, else the face itself.
    pub fn enhance_out(&mut self, g: &Buffer, band: Option<(&Buffer, usize)>, c: &Act) -> Result<Buffer, String> {
        let n = c.w;
        let out = self.buffer(n * n * 3 * 4, false)?;
        let (d, lo) = band.unwrap_or((g, 0));
        let p = BandParams { n: n as u32, lo: lo as u32, k: 0, s: 0.0 };
        self.elementwise(&self.ops.restore()?.enhance_out, &[g, d, &c.buf, &out], &p, n * n);
        self.done("enhance out")?;
        Ok(out)
    }

    // ------------------------------------------------------------------ the occlusion mask

    /// FRN's statistics of x (`groups` > 1: a polyphase transposed conv's), with the layer's
    /// [gamma, beta, tau] and epsilon: float4 (scale, shift, floor) per channel.
    pub fn frn_stats(&mut self, x: &Act, frn: &Buffer, eps: f32, groups: usize) -> Result<Buffer, String> {
        let c = x.c / groups;
        let out = self.buffer(c * 16, false)?;
        let p = FrnParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: c as u32, groups: groups as u32, eps };
        let tg = MTLSize { width: c.div_ceil(32), height: 1, depth: 1 };
        self.dispatch(&self.ops.restore()?.frn_stats, &[(&x.buf, 0), (frn, 0), (&out, 0)], &p, tg, MTLSize { width: 32, height: 32, depth: 1 });
        self.done("frn stats")?;
        Ok(out)
    }

    /// FRN + TLU with frn_stats' values (c channels); `shuffle`: x is a polyphase transposed conv
    /// (the output twice its size); `skip`: a compact map of the output's size appended.
    pub fn frn_out(&mut self, x: &Act, pro: &Buffer, c: usize, shuffle: bool, skip: Option<&Act>) -> Result<Act, String> {
        let (oh, ow) = if shuffle { (2 * x.h, 2 * x.w) } else { (x.h, x.w) };
        let cs = skip.map_or(0, |s| s.c);
        if let Some(s) = skip {
            if (s.h, s.w, s.sw) != (oh, ow, ow) {
                return Err(format!("a {}x{} map joined to a {oh}x{ow} one", s.w, s.h));
            }
        }
        let out = self.buffer(oh * ow * (c + cs) * 2, false)?;
        let p = FrnOutParams { oh: oh as u32, ow: ow as u32, c: c as u32, cs: cs as u32, xsw: x.sw as u32, shuffle: shuffle as u32 };
        let sk = skip.map(|s| s.buf.clone()).unwrap_or_else(|| x.buf.clone());
        self.elementwise(&self.ops.restore()?.frn_out, &[&x.buf, pro, &sk, &out], &p, oh * ow * (c + cs));
        self.done("frn")?;
        Ok(Act { buf: out, h: oh, w: ow, sw: ow, c: c + cs })
    }

    /// XSeg's blurred halving with the separable filter `taps` (3 or 4, normalized here).
    pub fn blur_down(&mut self, x: &Act, taps: &[f32]) -> Result<Act, String> {
        let (oh, ow) = (x.h / 2, x.w / 2);
        let out = self.buffer(oh * ow * x.c * 2, false)?;
        let sum: f32 = taps.iter().sum();
        let mut k = [0f32; 4];
        k.iter_mut().zip(taps).for_each(|(d, t)| *d = t / sum);
        let p = BlurDownParams { h: x.h as u32, w: x.w as u32, sw: x.sw as u32, c: x.c as u32, taps: taps.len() as u32, k };
        self.elementwise(&self.ops.restore()?.blur_down, &[&x.buf, &out], &p, oh * ow * x.c);
        self.done("blur down")?;
        Ok(Act { buf: out, h: oh, w: ow, sw: ow, c: x.c })
    }

    /// An f32 vector as an [h, w, c] activation.
    pub fn to_half(&mut self, x: &Buffer, h: usize, w: usize, c: usize) -> Result<Act, String> {
        let n = h * w * c;
        let out = self.buffer(n * 2, false)?;
        self.elementwise(&self.ops.restore()?.to_half, &[x, &out], &(n as u32), n);
        self.done("to half")?;
        Ok(Act { buf: out, h, w, sw: w, c })
    }

    /// The occlusion mask from XSeg's output (f32 [n, n]): grown by `grow` px (shrunk when
    /// negative), blurred with `taps` (f32, 2r + 1), then (clamp(m, 0.5, 1) - 0.5) * 2.
    pub fn occlusion_mask(&mut self, y: &Buffer, n: usize, grow: isize, taps: &Buffer, r: usize) -> Result<Buffer, String> {
        let src = if grow != 0 {
            let g = self.buffer(n * n * 4, false)?;
            let p = MaskParams { n: n as u32, r: grow.unsigned_abs() as u32, erode: (grow < 0) as u32, vertical: 0, finish: 0 };
            self.elementwise(&self.ops.restore()?.mask_morph, &[y, &g], &p, n * n);
            self.done("mask grow")?;
            g
        } else {
            y.clone()
        };
        let h = self.buffer(n * n * 4, false)?;
        let p = MaskParams { n: n as u32, r: r as u32, erode: 0, vertical: 0, finish: 0 };
        self.elementwise(&self.ops.restore()?.mask_blur, &[&src, taps, &h], &p, n * n);
        self.done("mask blur")?;
        let out = self.buffer(n * n * 4, false)?;
        let p = MaskParams { n: n as u32, r: r as u32, erode: 0, vertical: 1, finish: 1 };
        self.elementwise(&self.ops.restore()?.mask_blur, &[&h, taps, &out], &p, n * n);
        self.done("mask blur")?;
        Ok(out)
    }

    /// Puts `orig` back into `frame` (in place, the same size) where the occlusion mask (f32
    /// [n, n], through `m`: frame pixel -> mask pixel) says something covers the face.
    pub fn unocclude(&mut self, frame: &Img8, orig: &Img8, mask: &Buffer, n: usize, m: &Affine) -> Result<(), String> {
        if (frame.w, frame.h) != (orig.w, orig.h) {
            return Err("unocclude: frames of different sizes".into());
        }
        let [x1, y1, x2, y2] = crate::photo::paste_box(m, n, frame.w, frame.h);
        if x2 <= x1 || y2 <= y1 {
            return Ok(());
        }
        let p = UnoccludeParams {
            m: m.map(|v| v as f32),
            x1: x1 as u32,
            y1: y1 as u32,
            bw: (x2 - x1) as u32,
            bh: (y2 - y1) as u32,
            stride: frame.stride as u32,
            bgra: frame.bgra as u32,
            ostride: orig.stride as u32,
            obgra: orig.bgra as u32,
            n: n as u32,
        };
        self.elementwise(&self.ops.restore()?.unocclude, &[&frame.buf, &orig.buf, mask], &p, (x2 - x1) * (y2 - y1));
        self.done("unocclude")
    }
}

/// What a graph run gave: the result, the GPU time in ms, and in profile mode each step's time.
pub(super) struct Ran<T> {
    pub value: T,
    pub ms: f64,
    pub steps: Vec<(&'static str, f64)>,
}

/// Runs `build` and waits. In profile mode each step runs in its own command buffer.
pub(super) fn run<T>(kernels: &mut Kernels, ops: &Ops, profile: bool, build: impl FnOnce(&mut Graph) -> Result<T, String>) -> Result<Ran<T>, String> {
    run_in(kernels, ops, None, profile, build)
}

/// `run`, taking its buffers from `arena` when given one.
pub(super) fn run_in<T>(kernels: &mut Kernels, ops: &Ops, arena: Option<&mut Arena>, profile: bool, build: impl FnOnce(&mut Graph) -> Result<T, String>) -> Result<Ran<T>, String> {
    // what Metal autoreleases is let go of here, not whenever the calling thread's pool drains
    objc2::rc::autoreleasepool(|_| run_now(kernels, ops, arena, profile, build))
}

fn run_now<T>(kernels: &mut Kernels, ops: &Ops, arena: Option<&mut Arena>, profile: bool, build: impl FnOnce(&mut Graph) -> Result<T, String>) -> Result<Ran<T>, String> {
    let queue = kernels.queue.clone();
    let commands = queue.commandBuffer().ok_or("no command buffer")?;
    let encoder = commands.computeCommandEncoder().ok_or("no encoder")?;
    let mut g = Graph { kernels, ops, queue, commands, encoder, keep: Vec::new(), arena, next: 0, profile: profile.then(Vec::new) };
    let result = build(&mut g);
    g.encoder.endEncoding();
    let value = result?;
    g.commands.commit();
    g.commands.waitUntilCompleted();
    if let Some(e) = g.commands.error() {
        return Err(e.localizedDescription().to_string());
    }
    match g.profile {
        Some(steps) => Ok(Ran { value, ms: steps.iter().map(|s| s.1).sum(), steps }),
        None => Ok(Ran { value, ms: (g.commands.GPUEndTime() - g.commands.GPUStartTime()) * 1e3, steps: Vec::new() }),
    }
}
