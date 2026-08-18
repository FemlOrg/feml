//! Backend abstraction (ggml-backend inspired, Rust-flavored).
//!
//! Layer mapping to ggml:
//! - `BackendRegistrar` ~ `ggml_backend_reg` (plugin entry, discovery, scoring)
//! - `BackendDevice`    ~ `ggml_backend_dev`  (device info, capability queries)
//! - `Backend`          ~ `ggml_backend`      (execution stream, buffer creation)
//!
//! Buffers are accessed through opaque `BufferHandle`s owned by the backend
//! (ggml's `ggml_backend_buffer` model). This keeps the public API free of
//! `Any`/downcasting and lets graph plans store plain handles.
//!
//! `BufferType`/`BackendBuffer` (the remaining two ggml layers) land in M1
//! together with the memory planner that needs buffer metadata.

use crate::dtype::DType;
use crate::error::Result;
use crate::op::Op;
use crate::plan::GraphPlan;
use std::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DeviceType {
    #[default]
    Cpu,
    Gpu,
    Accelerator,
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// Backend supports asynchronous tensor IO + synchronize.
    pub async_compute: bool,
    /// Backend can wrap host (pinned) memory.
    pub host_buffer: bool,
    /// Backend supports event-based multi-stream sync.
    pub events: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub memory_total: usize,
    pub memory_free: usize,
    pub device_type: DeviceType,
    pub caps: Capabilities,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferUsage {
    /// Long-lived weights/constants (never freed by the planner).
    Weights,
    /// Transient compute/workspace memory (reusable by the planner).
    Compute,
}

/// Opaque handle to a backend-owned buffer (backend-scoped).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferHandle(u32);

impl BufferHandle {
    /// Backend-internal: create a handle. Backend implementations (including
    /// third-party crates) use this to hand out buffer identifiers.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl fmt::Display for BufferHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "buf#{}", self.0)
    }
}

/// Execution stream: buffer lifecycle, compute, synchronization.
///
/// Implementations must be `Send + Sync` (plans execute on worker threads).
/// Buffer handles are backend-scoped: a handle is only valid on the backend
/// that created it.
pub trait Backend: Send + Sync {
    /// Stable, human-readable backend identifier ("CPU", "OpenCL", ...).
    fn name(&self) -> &str;

    /// Allocate a zero-initialized buffer of `size` bytes. `usage` hints the
    /// planner (weights are never freed; compute memory is reusable).
    fn create_buffer(&self, size: usize, usage: BufferUsage) -> Result<BufferHandle>;

    /// Free a buffer. The handle becomes invalid.
    fn release_buffer(&self, handle: BufferHandle) -> Result<()>;

    /// Copy `data` into the buffer at `offset`. The byte range
    /// `[offset, offset + data.len())` must lie within the buffer; violations
    /// return an error (never panic, never write out of bounds).
    fn write(&self, handle: BufferHandle, offset: usize, data: &[u8]) -> Result<()>;

    /// Copy `out.len()` bytes from the buffer at `offset` into `out`.
    fn read(&self, handle: BufferHandle, offset: usize, out: &mut [u8]) -> Result<()>;

    /// Fill `len` bytes with `value` starting at `offset`.
    fn fill(&self, handle: BufferHandle, offset: usize, value: u8, len: usize) -> Result<()>;

    /// Wait for all previously enqueued work to complete (no-op on CPU).
    fn synchronize(&self) -> Result<()>;

    /// Execute a compiled plan. The plan is static: all buffer handles,
    /// offsets and kernels were resolved at compile time. Execution must
    /// perform no allocation and no user-visible panic.
    fn graph_compute(&self, plan: &GraphPlan) -> Result<()>;

    /// Whether this backend can execute `op` for the given source dtypes.
    /// Ops not reported here are routed elsewhere (fallback backends).
    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        let _ = (op, src_dtypes);
        false
    }
}

/// Device: structured info + capability queries.
///
/// A device represents one physical compute resource. `init_backend` may be
/// called multiple times; each call returns an independent execution stream.
pub trait BackendDevice: Send + Sync {
    /// Structured device information (never derived from name matching).
    fn info(&self) -> Result<DeviceInfo>;

    /// Create a fresh execution stream bound to this device.
    fn init_backend(&self) -> Result<Box<dyn Backend>>;

    /// Same contract as `Backend::supports_op`.
    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool;

    /// Whether an expensive op would be better off on this device
    /// (used by the scheduler for heterogeneous placement).
    fn offload_op(&self, op: Op) -> bool {
        let _ = op;
        false
    }

    /// Wrap an externally owned host pointer (mmap models, interop) into a
    /// backend buffer. Implementations that cannot support this return an
    /// error; the pointer must outlive the returned buffer.
    fn buffer_from_host_ptr(
        &self,
        ptr: &mut [u8],
        size: usize,
        max_tensor_size: usize,
    ) -> Result<BufferHandle>;
}

/// Backend plugin entry point: discovery + scoring for `open_best`.
pub trait BackendRegistrar: Send + Sync {
    fn name(&self) -> &str;
    fn device_count(&self) -> usize;
    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>>;
    /// Higher = preferred. 0 = unavailable in this environment.
    fn score(&self) -> u32 {
        0
    }
}
