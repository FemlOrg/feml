//! CUDA execution stream: buffer lifecycle, kernel dispatch, sync.
//!
//! All state lives behind `Arc<Mutex<_>>` so the backend is `Send + Sync`.
//! Kernel sources are compiled to PTX once at init (NVRTC, targeting the
//! device's compute capability); buffers are `CudaSlice<u8>` owned by the
//! backend. Everything executes on one stream; `synchronize` flushes it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};

use feml_core::backend::{Backend, BufferHandle, BufferUsage};
use feml_core::dtype::DType;
use feml_core::error::{Error, Result};
use feml_core::op::Op;
use feml_core::plan::GraphPlan;

use crate::device::cuda_err;

pub(crate) struct BackendInner {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) funcs: HashMap<&'static str, CudaFunction>,
    pub(crate) buffers: HashMap<BufferHandle, CudaSlice<u8>>,
    next_handle: u32,
}

#[derive(Clone)]
pub struct CudaBackend {
    pub(crate) inner: Arc<Mutex<BackendInner>>,
}

impl CudaBackend {
    pub(crate) fn new(ctx: Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();

        // Compile the embedded kernel source for this device's compute
        // capability (NVRTC default targets an old arch that modern drivers
        // may refuse to JIT). FMA contraction is disabled so accumulation
        // order matches the CPU backend bit-for-bit.
        let (major, minor) = ctx.compute_capability().map_err(cuda_err)?;
        let opts = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            fmad: Some(false),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(include_str!("kernels.cu"), opts)
            .map_err(|e: CompileError| Error::backend("CUDA", format!("kernel compile: {e}")))?;
        let module = ctx.load_module(ptx).map_err(cuda_err)?;

        let mut funcs = HashMap::new();
        for name in ["kernel_mul", "kernel_add", "kernel_mul_mat", "kernel_fill"] {
            let func = module.load_function(name).map_err(cuda_err)?;
            funcs.insert(name, func);
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(BackendInner {
                stream,
                funcs,
                buffers: HashMap::new(),
                next_handle: 1,
            })),
        })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn create_buffer(&self, size: usize, usage: BufferUsage) -> Result<BufferHandle> {
        let mut inner = self.inner.lock().unwrap();
        let BackendInner { stream, buffers, next_handle, .. } = &mut *inner;
        let slice = stream
            .alloc_zeros::<u8>(size)
            .map_err(|e: DriverError| Error::backend("CUDA", format!("create_buffer: {e}")))?;
        let handle = BufferHandle::new(*next_handle);
        *next_handle += 1;
        buffers.insert(handle, slice);
        let _ = usage; // usage metadata arrives with the planner (M1)
        Ok(handle)
    }

    fn release_buffer(&self, handle: BufferHandle) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .buffers
            .remove(&handle)
            .ok_or_else(|| Error::msg(format!("release_buffer: unknown buffer {handle}")))?;
        Ok(())
    }

    fn write(&self, handle: BufferHandle, offset: usize, data: &[u8]) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        let BackendInner { stream, buffers, .. } = &mut *inner;
        let slice = buffers
            .get_mut(&handle)
            .ok_or_else(|| Error::msg(format!("write: unknown buffer {handle}")))?;
        let end =
            offset.checked_add(data.len()).ok_or_else(|| Error::msg("write: offset overflow"))?;
        if end > slice.num_bytes() {
            return Err(Error::msg(format!(
                "write: {handle} overflow: offset {offset} + {} > buffer {}",
                data.len(),
                slice.num_bytes()
            )));
        }
        let mut view = slice.try_slice_mut(offset..end).ok_or_else(|| {
            Error::msg(format!("write: {handle} range {offset}..{end} out of bounds"))
        })?;
        // Blocking, like the CPU/OpenCL backends.
        stream.memcpy_htod(data, &mut view).map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)
    }

    fn read(&self, handle: BufferHandle, offset: usize, out: &mut [u8]) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let BackendInner { stream, buffers, .. } = &*inner;
        let slice = buffers
            .get(&handle)
            .ok_or_else(|| Error::msg(format!("read: unknown buffer {handle}")))?;
        let end =
            offset.checked_add(out.len()).ok_or_else(|| Error::msg("read: offset overflow"))?;
        if end > slice.num_bytes() {
            return Err(Error::msg(format!(
                "read: {handle} overflow: offset {offset} + {} > buffer {}",
                out.len(),
                slice.num_bytes()
            )));
        }
        let view = slice.try_slice(offset..end).ok_or_else(|| {
            Error::msg(format!("read: {handle} range {offset}..{end} out of bounds"))
        })?;
        // Blocking, like the CPU/OpenCL backends: the contract lets callers
        // use `out` immediately after `read` returns.
        stream.memcpy_dtoh(&view, out).map_err(cuda_err)?;
        stream.synchronize().map_err(cuda_err)
    }

    fn fill(&self, handle: BufferHandle, offset: usize, value: u8, len: usize) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let slice = inner
            .buffers
            .get(&handle)
            .ok_or_else(|| Error::msg(format!("fill: unknown buffer {handle}")))?;
        if len == 0 {
            return Ok(());
        }
        let end = offset.checked_add(len).ok_or_else(|| Error::msg("fill: offset overflow"))?;
        if end > slice.num_bytes() {
            return Err(Error::msg(format!(
                "fill: {handle} overflow: offset {offset} + {len} > buffer {}",
                slice.num_bytes()
            )));
        }
        super::ops::fill::fill(&inner, slice, offset, value, len)
    }

    fn synchronize(&self) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        inner.stream.synchronize().map_err(cuda_err)
    }

    fn graph_compute(&self, plan: &GraphPlan) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        for &ti in &plan.nodes {
            let dst = &plan.tensors[ti];
            let sa = &plan.tensors[dst.srcs[0]];
            let sb = &plan.tensors[dst.srcs[1]];
            match dst.op {
                Op::Mul | Op::Add => {
                    super::ops::elementwise::elementwise_binary(
                        &inner,
                        plan,
                        sa,
                        sb,
                        dst,
                        dst.op == Op::Mul,
                    )?;
                }
                Op::MulMat => {
                    super::ops::mul_mat::mul_mat(&inner, plan, sa, sb, dst)?;
                }
                _ => {
                    return Err(Error::msg(format!("CUDA: op {} not implemented", dst.op.name())));
                }
            }
        }
        Ok(())
    }

    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        matches!(op, Op::Mul | Op::Add | Op::MulMat) && src_dtypes.iter().all(|d| *d == DType::F32)
    }
}
