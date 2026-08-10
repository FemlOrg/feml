//! feml-opencl: OpenCL backend for feml-core.
//!
//! Kernel sources are embedded at compile time (`include_str!`); buffers are
//! opaque handles managed by the backend (see `feml-core::backend::Backend`).

mod backend;
mod device;
mod ops;

pub use backend::OpenclBackend;
pub use device::{OpenclBackendDevice, OpenclRegistrar};

/// Versioned plugin entry points (ggml `GGML_BACKEND_API_VERSION` analog).
pub const API_VERSION: u32 = 1;
