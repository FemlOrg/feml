//! feml-cpu: CPU backend for feml-core.
//!
//! Mirrors the feml-opencl structure: opaque buffer handles, no panics,
//! all operations return `Result`.

mod backend;
mod device;
mod ops;

pub use backend::CpuBackend;
pub use device::{CpuBackendDevice, CpuRegistrar};

/// Versioned plugin entry points (ggml `GGML_BACKEND_API_VERSION` analog).
pub const API_VERSION: u32 = 1;
