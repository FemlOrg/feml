//! OpenCL execution stream: buffer lifecycle, kernel dispatch, sync.
//!
//! All state lives behind `Arc<Mutex<_>>` so the backend is `Send + Sync`
//! (M1 replaces the coarse lock with per-stream queues).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use feml_core::backend::{Backend, BufferHandle, BufferUsage};
use feml_core::dtype::DType;
use feml_core::error::{Error, Result};
use feml_core::op::Op;
use feml_core::plan::GraphPlan;

pub(crate) struct BackendInner {
    pub(crate) queue: ocl::Queue,
    pub(crate) kernels: HashMap<&'static str, ocl::Kernel>,
    pub(crate) buffers: HashMap<BufferHandle, ocl::Buffer<u8>>,
    next_handle: u32,
}

#[derive(Clone)]
pub struct OpenclBackend {
    pub(crate) inner: Arc<Mutex<BackendInner>>,
}

impl OpenclBackend {
    fn get_buffer(
        inner: &BackendInner,
        plan: &GraphPlan,
        t: &feml_core::plan::PlanTensor,
    ) -> Result<ocl::Buffer<u8>> {
        let h = plan.buffers[t.buffer].handle;
        inner
            .buffers
            .get(&h)
            .cloned()
            .ok_or_else(|| Error::msg(format!("graph_compute: unknown buffer {h}")))
    }

    fn ocl_err(e: ocl::Error) -> Error {
        Error::backend("OpenCL", e.to_string())
    }

    pub fn new(device: ocl::Device, platform: ocl::Platform) -> Result<Self> {
        let devices = vec![device];
        let context = ocl::Context::builder()
            .platform(platform)
            .devices(&devices)
            .build()
            .map_err(Self::ocl_err)?;
        let queue = ocl::Queue::new(&context, device, None).map_err(Self::ocl_err)?;

        let mut kernels = HashMap::new();
        let program = ocl::Program::builder()
            .src(include_str!("kernels/mul.cl"))
            .devices(device)
            .build(&context)
            .map_err(Self::ocl_err)?;
        let kernel_mul = ocl::Kernel::builder()
            .program(&program)
            .name("kernel_mul")
            .build()
            .map_err(Self::ocl_err)?;
        kernels.insert("kernel_mul", kernel_mul);

        let program_mm = ocl::Program::builder()
            .src(include_str!("kernels/mul_mat.cl"))
            .devices(device)
            .build(&context)
            .map_err(Self::ocl_err)?;
        let kernel_mul_mat = ocl::Kernel::builder()
            .program(&program_mm)
            .name("kernel_mul_mat")
            .build()
            .map_err(Self::ocl_err)?;
        kernels.insert("kernel_mul_mat", kernel_mul_mat);

        Ok(Self {
            inner: Arc::new(Mutex::new(BackendInner {
                queue,
                kernels,
                buffers: HashMap::new(),
                next_handle: 1,
            })),
        })
    }
}

impl Backend for OpenclBackend {
    fn name(&self) -> &str {
        "OpenCL"
    }

    fn create_buffer(&self, size: usize, usage: BufferUsage) -> Result<BufferHandle> {
        let mut inner = self.inner.lock().unwrap();
        let buffer = ocl::Buffer::<u8>::builder()
            .queue(inner.queue.clone())
            .len(size)
            .build()
            .map_err(|e| Error::backend("OpenCL", format!("create_buffer: {e}")))?;
        let handle = BufferHandle::new(inner.next_handle);
        inner.next_handle += 1;
        inner.buffers.insert(handle, buffer);
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
        let inner = self.inner.lock().unwrap();
        let buffer = inner
            .buffers
            .get(&handle)
            .ok_or_else(|| Error::msg(format!("write: unknown buffer {handle}")))?;
        if offset + data.len() > buffer.len() {
            return Err(Error::msg(format!(
                "write: {handle} overflow: offset {offset} + {} > buffer {}",
                data.len(),
                buffer.len()
            )));
        }
        unsafe {
            ocl::core::enqueue_write_buffer(
                &inner.queue,
                buffer,
                true, // blocking
                offset,
                data,
                None::<ocl::core::Event>,
                None::<()>,
            )
        }
        .map_err(|e| Error::backend("OpenCL", format!("write: {e}")))
    }

    fn read(&self, handle: BufferHandle, offset: usize, out: &mut [u8]) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let buffer = inner
            .buffers
            .get(&handle)
            .ok_or_else(|| Error::msg(format!("read: unknown buffer {handle}")))?;
        if offset + out.len() > buffer.len() {
            return Err(Error::msg(format!(
                "read: {handle} overflow: offset {offset} + {} > buffer {}",
                out.len(),
                buffer.len()
            )));
        }
        unsafe {
            ocl::core::enqueue_read_buffer(
                &inner.queue,
                buffer,
                true, // blocking
                offset,
                out,
                None::<ocl::core::Event>,
                None::<()>,
            )
        }
        .map_err(|e| Error::backend("OpenCL", format!("read: {e}")))
    }

    fn fill(&self, handle: BufferHandle, offset: usize, value: u8, len: usize) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let buffer = inner
            .buffers
            .get(&handle)
            .ok_or_else(|| Error::msg(format!("fill: unknown buffer {handle}")))?;
        if offset + len > buffer.len() {
            return Err(Error::msg(format!(
                "fill: {handle} overflow: offset {offset} + {len} > buffer {}",
                buffer.len()
            )));
        }
        ocl::core::enqueue_fill_buffer(
            &inner.queue,
            buffer,
            value,
            offset,
            len,
            None::<ocl::core::Event>,
            None::<()>,
            None,
        )
        .map_err(|e| Error::backend("OpenCL", format!("fill: {e}")))
    }

    fn synchronize(&self) -> Result<()> {
        let inner = self.inner.lock().unwrap();
        let event = inner
            .queue
            .enqueue_marker(None::<()>)
            .map_err(|e| Error::backend("OpenCL", format!("synchronize: {e}")))?;
        event.wait_for().map_err(|e| Error::backend("OpenCL", format!("synchronize: {e}")))
    }

    fn graph_compute(&self, plan: &GraphPlan) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        for &ti in &plan.nodes {
            let dst = &plan.tensors[ti];
            let sa = &plan.tensors[dst.srcs[0]];
            let sb = &plan.tensors[dst.srcs[1]];
            match dst.op {
                Op::Mul => {
                    let a = super::ops::mul::TensorArg {
                        buf: &Self::get_buffer(&inner, plan, sa)?,
                        offset: sa.offset,
                        shape: &sa.layout.shape,
                        stride: &sa.layout.stride,
                    };
                    let b = super::ops::mul::TensorArg {
                        buf: &Self::get_buffer(&inner, plan, sb)?,
                        offset: sb.offset,
                        shape: &sb.layout.shape,
                        stride: &sb.layout.stride,
                    };
                    let c = super::ops::mul::TensorArg {
                        buf: &Self::get_buffer(&inner, plan, dst)?,
                        offset: dst.offset,
                        shape: &dst.layout.shape,
                        stride: &dst.layout.stride,
                    };
                    super::ops::mul::mul(&mut inner, &a, &b, &c)?;
                }
                Op::MulMat => {
                    let a = super::ops::mul_mat::MatArg {
                        buf: &Self::get_buffer(&inner, plan, sa)?,
                        offset: sa.offset,
                        stride1: sa.layout.stride[1],
                    };
                    let b = super::ops::mul_mat::MatArg {
                        buf: &Self::get_buffer(&inner, plan, sb)?,
                        offset: sb.offset,
                        stride1: sb.layout.stride[1],
                    };
                    let c = super::ops::mul_mat::MatArg {
                        buf: &Self::get_buffer(&inner, plan, dst)?,
                        offset: dst.offset,
                        stride1: dst.layout.stride[1],
                    };
                    super::ops::mul_mat::mul_mat(
                        &mut inner,
                        &a,
                        &b,
                        &c,
                        sa.layout.shape[0],
                        sa.layout.shape[1],
                        sb.layout.shape[1],
                    )?;
                }
                _ => {
                    return Err(Error::msg(format!(
                        "OpenCL: op {} not implemented",
                        dst.op.name()
                    )));
                }
            }
        }
        Ok(())
    }

    fn supports_op(&self, op: Op, src_dtypes: &[DType]) -> bool {
        matches!(op, Op::Mul) && src_dtypes.iter().all(|d| *d == DType::F32)
    }
}
