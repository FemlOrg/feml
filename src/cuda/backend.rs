use super::backend_buffer::CudaBackendBuffer;
use super::backend_context::CudaBackendContext;
use super::backend_register::CudaBackendRegister;
use crate::backend::BackendBufferUsage;
use crate::backend::{Backend, BackendBuffer, BackendRegister};
use crate::compute_graph::ComputeGraph;
use crate::context::Context;
use crate::cuda::kernels::mul::mul;
use crate::data_type::TensorOpType;
use crate::error::{Error, ErrorKind, Result};
use crate::tensor::Tensor;
use cuda_core::DeviceBuffer;
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct CudaBackend {
    pub(super) backend_ctx: Rc<RefCell<CudaBackendContext>>,
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn synchronize(&self) -> Result<()> {
        let stream = self.backend_ctx.borrow_mut().ensure_current_stream()?;
        stream.synchronize()?;
        Ok(())
    }

    fn graph_compute(&self, ctx: &Context, graph: &mut ComputeGraph) -> Result<()> {
        for node in graph.nodes().iter() {
            let tensor = ctx.get_tensor(*node)?;
            self.compute_forward(ctx, &tensor)?;
        }

        Ok(())
    }

    fn write_async(
        &self,
        _tensor: Tensor,
        _data: &mut [u8],
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        todo!()
    }

    fn read_async(
        &self,
        _tensor: Tensor,
        _data: &mut [u8],
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        todo!()
    }

    fn copy_async(&self, src: Tensor, dst: Tensor) -> Result<()> {
        let src_storage = &*src.storage()?;
        let dst_storage = &*dst.storage()?;

        let src_buffer =
            src_storage.as_cuda().ok_or_else(|| Error::msg("src tensor storage is not CUDA"))?;
        let dst_buffer =
            dst_storage.as_cuda().ok_or_else(|| Error::msg("dst tensor storage is not CUDA"))?;

        let src_backend_ctx = src_buffer
            .backend_ctx
            .as_ref()
            .ok_or_else(|| Error::msg("src cuda backend context is missing"))?;

        let dst_backend_ctx = dst_buffer
            .backend_ctx
            .as_ref()
            .ok_or_else(|| Error::msg("dst cuda backend context is missing"))?;

        let src_offset = src_storage.offset();
        let dst_offset = dst_storage.offset();
        let src_ptr = src_buffer.buffer.cu_deviceptr() + src_offset as u64;
        let dst_ptr = dst_buffer.buffer.cu_deviceptr() + dst_offset as u64;
        let src_bytes = src.nbytes();

        let stream = src_backend_ctx.borrow_mut().ensure_current_stream()?;
        let same_device = src_backend_ctx.borrow().current_device_id
            == dst_backend_ctx.borrow().current_device_id;

        unsafe {
            if same_device {
                cuda_bindings::cuMemcpyDtoDAsync_v2(
                    dst_ptr,
                    src_ptr,
                    src_bytes,
                    stream.cu_stream(),
                );
            } else {
                let src_cuda_ctx = src_backend_ctx
                    .borrow()
                    .current_context
                    .as_ref()
                    .ok_or_else(|| Error::msg("src cuda context is missing"))?
                    .cu_ctx();
                let dst_cuda_ctx = dst_backend_ctx
                    .borrow()
                    .current_context
                    .as_ref()
                    .ok_or_else(|| Error::msg("dst cuda context is missing"))?
                    .cu_ctx();

                cuda_bindings::cuMemcpyPeerAsync(
                    dst_ptr,
                    dst_cuda_ctx,
                    src_ptr,
                    src_cuda_ctx,
                    src_bytes,
                    stream.cu_stream(),
                );
            }
        }

        Ok(())
    }

    fn create_buffer(
        &self,
        size: usize,
        usage: BackendBufferUsage,
    ) -> Result<Box<dyn BackendBuffer>> {
        let stream = self.backend_ctx.borrow_mut().ensure_current_stream()?;
        DeviceBuffer::<u8>::zeroed(&stream, size)
            .map(|buffer| {
                Box::new(CudaBackendBuffer::new(
                    Some(self.backend_ctx.clone()),
                    buffer,
                    usage,
                    size,
                )) as Box<dyn BackendBuffer>
            })
            .map_err(|e| Error::msg(format!("Failed to allocate CUDA DeviceBuffer: {}", e)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl CudaBackend {
    pub fn init(device_id: i32) -> Result<Box<dyn Backend>> {
        let reg =
            CudaBackendRegister::init().as_any().downcast_ref::<CudaBackendRegister>().unwrap();
        let device = reg.device(device_id as usize)?;
        let cuda_device = device
            .as_any()
            .downcast_ref::<super::backend_device::CudaBackendDevice>()
            .ok_or_else(|| Error::msg("device is not a CUDA device"))?;

        Ok(Box::new(Self {
            backend_ctx: cuda_device
                .backend_ctx
                .clone()
                .ok_or_else(|| Error::msg("backend_ctx is none"))?,
        }))
    }

    fn compute_forward(&self, ctx: &Context, tensor: &Tensor) -> Result<()> {
        let src_tensor = tensor.src_tensor();
        match tensor.op_type() {
            TensorOpType::TensorOpMul => {
                let src0 = ctx.get_tensor(src_tensor[0])?;
                let src1 = ctx.get_tensor(src_tensor[1])?;
                mul(self, &src0, &src1, tensor)
            }
            _ => {
                let op_name = format!("{:?}", tensor.op_type());
                Err(Error::new(ErrorKind::UnsupportedBackendOp {
                    backend: "cuda",
                    op: "compute_forward",
                })
                .context(format!("unsupported op type: {}", op_name))
                .context("in CudaBackend::compute_forward"))
            }
        }
    }
}
