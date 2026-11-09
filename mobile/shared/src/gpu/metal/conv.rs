//! The conv kernels, compiled per tiling.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice, MTLLanguageVersion, MTLResourceOptions};

use super::{compile, Device, Pipeline};

const TENSOR: &str = include_str!("conv_tensor.metal");
pub(super) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(super) type Queue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

#[repr(C)]
pub(super) struct TensorParams {
    pub in_w: u32,
    pub out_h: u32,
    pub out_w: u32,
    pub m: u32,
    pub cin: u32,
    pub cout: u32,
    pub k: u32,
    pub stride: u32,
    pub slope: f32,
}

/// How a conv is tiled.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Variant {
    /// rows computed as wide as the padded input, so tiles run across rows (stride 1 only)
    pub wide: bool,
    pub bm: usize,
    pub bn: usize,
    /// threadgroups the taps are split over
    pub split: usize,
}

impl Variant {
    /// The default for a conv of this stride, output width and output channels.
    pub(super) fn for_shape(stride: usize, out_w: usize, cout: usize) -> Variant {
        let bn = if cout >= 128 { 128 } else if cout >= 64 { 64 } else { 32 };
        // wide tiles double the speed of the 32-pixel style rows, but are
        // slower on rows of 64 pixels or more, and splitting the taps never helped
        if stride == 1 && out_w < 64 {
            Variant { wide: true, bm: 64, bn, split: 1 }
        } else {
            Variant { wide: false, bm: if out_w >= 64 { 64 } else { 32 }, bn, split: 1 }
        }
    }
}

pub(super) struct Kernels {
    pub device: Retained<Device>,
    pub queue: Queue,
    /// the conv kernel and, when split, its reduce kernel
    pipelines: HashMap<Variant, (Pipeline, Option<Pipeline>)>,
}

impl Kernels {
    pub(super) fn new(device: Retained<Device>, queue: Queue) -> Kernels {
        Kernels { device, queue, pipelines: HashMap::new() }
    }

    pub(super) fn pipelines(&mut self, v: Variant) -> Result<(Pipeline, Option<Pipeline>), String> {
        if let Some(p) = self.pipelines.get(&v) {
            return Ok(p.clone());
        }
        let src = format!("#define BM {}\n#define BN {}\n#define WIDE {}\n#define SPLIT {}\n{TENSOR}", v.bm, v.bn, v.wide as u8, v.split);
        let v4 = Some(MTLLanguageVersion::Version4_0);
        let conv = compile(&self.device, &src, "conv", v4)?;
        let reduce = if v.split > 1 { Some(compile(&self.device, &src, "reduce", v4)?) } else { None };
        self.pipelines.insert(v, (conv.clone(), reduce.clone()));
        Ok((conv, reduce))
    }
}

pub(super) fn buffer_with<T: Copy>(device: &Device, data: &[T]) -> Result<Buffer, String> {
    let len = std::mem::size_of_val(data);
    let ptr = NonNull::new(data.as_ptr() as *mut c_void).ok_or("empty data")?;
    // SAFETY: ptr points at len readable bytes, which Metal copies.
    unsafe { device.newBufferWithBytes_length_options(ptr, len, MTLResourceOptions::StorageModeShared) }
        .ok_or_else(|| "no buffer".into())
}
