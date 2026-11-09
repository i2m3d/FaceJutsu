//! The engine of the iOS app: finds and follows faces, and swaps them on the GPU (`session`, `scan`).

uniffi::setup_scaffolding!();

mod cpu;
mod enhancer;
mod gpu;
mod model;
mod occluder;
mod photo;
pub mod scan;
pub mod session;
mod track;

/// The engine's version, as set in Cargo.toml.
#[uniffi::export]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_matches_the_package() {
        assert_eq!(engine_version(), env!("CARGO_PKG_VERSION"));
    }
}
