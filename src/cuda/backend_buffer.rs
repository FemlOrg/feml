use std::cell::RefCell;
use std::rc::Rc;

use super::backend_context::CudaBackendContext;
use crate::backend::BackendBuffer;
use crate::backend::BackendBufferUsage;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use cuda_core::memory::{memcpy_dtoh_async, memcpy_htod_sync, memset_d8_async};
use cuda_core::DeviceBuffer;

pub struct CudaBackendBuffer {
    pub(super) backend_ctx: Option<Rc<RefCell<CudaBackendContext>>>,
    pub(super) buffer: DeviceBuffer<u8>,
    usage: BackendBufferUsage,
    size: usize,
}

impl CudaBackendBuffer {
    pub(super) fn new(
        backend_ctx: Option<Rc<RefCell<CudaBackendContext>>>,
        buffer: DeviceBuffer<u8>,
        usage: BackendBufferUsage,
        size: usize,
    ) -> Self {
        Self { backend_ctx, buffer, usage, size }
    }

    fn ctx(&self) -> Result<&Rc<RefCell<CudaBackendContext>>> {
        self.backend_ctx.as_ref().ok_or_else(|| Error::msg("backend_ctx is none"))
    }
}

impl BackendBuffer for CudaBackendBuffer {
    fn reset(&self) -> Result<()> {
        todo!()
    }

    fn init_tensor(&self, _tensor: Tensor, _offset: usize) -> Result<()> {
        todo!()
    }

    fn fill(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()> {
        let ctx = self.ctx()?;
        let mut ctx = ctx.borrow_mut();
        ctx.ensure_current_stream()?;
        let device_id = ctx.get_device_id()?;
        ctx.set_device(device_id)?;
        let stream =
            ctx.current_stream.clone().ok_or_else(|| Error::msg("current stream is none"))?;

        drop(ctx);

        let storage = &*tensor.storage()?;
        let _offset = offset + storage.offset();

        unsafe {
            let ptr = self.buffer.cu_deviceptr() as usize;
            memset_d8_async((ptr + _offset) as u64, value, size, stream.cu_stream())
                .map_err(|e| Error::msg(format!("memset_d8_async failed with error code {}", e)))?;

            stream.synchronize().map_err(|e| {
                Error::msg(format!("cuStreamSynchronize failed with error code {}", e))
            })?;
        }

        Ok(())
    }

    fn write(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize) -> Result<()> {
        let storage = &*tensor.storage()?;
        let _offset = offset + storage.offset();

        unsafe {
            let ptr = self.buffer.cu_deviceptr() as usize;
            memcpy_htod_sync(
                (ptr + _offset) as u64,
                data.as_ptr() as *const std::ffi::c_void,
                size,
            )
            .map_err(|e| Error::msg(format!("memcpy_htod_sync failed with error code {}", e)))?;
        }

        Ok(())
    }

    fn read(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize) -> Result<()> {
        let ctx = self.ctx()?;
        let stream = {
            let mut ctx = ctx.borrow_mut();
            ctx.ensure_current_stream()?;
            ctx.current_stream.clone().ok_or_else(|| Error::msg("current stream is none"))?
        };

        let storage = &*tensor.storage()?;
        let _offset = offset + storage.offset();

        unsafe {
            let ptr = self.buffer.cu_deviceptr() as usize;
            memcpy_dtoh_async(data.as_mut_ptr(), (ptr + _offset) as u64, size, stream.cu_stream())
                .map_err(|e| {
                    Error::msg(format!("memcpy_dtoh_async failed with error code {}", e))
                })?;

            stream.synchronize().map_err(|e| {
                Error::msg(format!("cuStreamSynchronize failed with error code {}", e))
            })?;
        }

        Ok(())
    }

    fn copy(&self, src: Tensor, dst: Tensor) -> Result<()> {
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

            stream.synchronize().map_err(|e| {
                Error::msg(format!("cuStreamSynchronize failed with error code {}", e))
            })?;
        }

        Ok(())
    }

    fn usage(&self) -> Result<BackendBufferUsage> {
        Ok(self.usage.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
