//! The GPU backend.

#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(target_vendor = "apple")]
pub(crate) use metal::{Enhance, Frame, Look, Session};
