//! The Metal backend.

mod conv;
mod graph;
mod nets;
mod pipeline;
mod restore;
mod session;
mod swapper;

pub(crate) use pipeline::{Enhance, Look};
pub(crate) use session::{Frame, Session};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLanguageVersion, MTLLibrary};

type Device = ProtocolObject<dyn MTLDevice>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// What running networks needs: the conv kernels (compiled per tiling, as used) and the small ops.
struct Gpu {
    kernels: conv::Kernels,
    ops: graph::Ops,
}

type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Compiles `src` on the device.
fn library(device: &Device, src: &str, version: Option<MTLLanguageVersion>) -> Result<Library, String> {
    let options = MTLCompileOptions::new();
    if let Some(v) = version {
        options.setLanguageVersion(v);
    }
    device
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&options))
        .map_err(|e| e.localizedDescription().to_string())
}

/// The kernel `name` from a compiled library.
fn pipeline(device: &Device, library: &Library, name: &str) -> Result<Pipeline, String> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or(format!("no function {name}"))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| e.localizedDescription().to_string())
}

/// Compiles `src` on the device and returns its kernel `name`.
fn compile(device: &Device, src: &str, name: &str, version: Option<MTLLanguageVersion>) -> Result<Pipeline, String> {
    pipeline(device, &library(device, src, version)?, name)
}
