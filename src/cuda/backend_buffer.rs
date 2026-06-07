use std::cell::RefCell;
use std::rc::Rc;

use super::backend_buffer_allocator::CudaBackendBufferAllocator;
use super::backend_context::CudaBackendContext;
use crate::backend::BackendBuffer;
use crate::backend::BackendBufferAllocator;
use crate::backend::BackendBufferUsage;
use crate::backend::BackendDevice;
use crate::data_type::is_quantized;
use crate::error::{Error, Result};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use cuda_core::memory::{memcpy_dtoh_async, memcpy_htod_async, memset_d8_async};
use cuda_core::DeviceBuffer;

pub(crate) struct CudaBackendBuffer {
    backend_ctx: Option<Rc<RefCell<CudaBackendContext>>>,
    backend_allocator: Option<Rc<CudaBackendBufferAllocator>>,
    buffer: DeviceBuffer<u8>,
    usage: BackendBufferUsage,
    size: usize,
}

impl BackendBuffer for CudaBackendBuffer {
    fn as_ptr(&self) -> Result<*mut u8> {
        todo!()
    }

    fn device(&self) -> Result<Box<dyn BackendDevice>> {
        todo!()
    }

    fn get_base(&self) -> Result<*mut u8> {
        todo!()
    }

    fn clear(&self, value: u8) -> Result<()> {
        todo!()
    }

    fn reset(&self) -> Result<()> {
        todo!()
    }

    fn init_tensor(&self, mut tensor: Tensor, offset: usize) -> Result<()> {
        if !is_quantized(tensor.get_dtype())
            || tensor.borrow().view_tensor.is_some()
            || self.usage == BackendBufferUsage::Compute
        {
            return Ok(());
        }

        let original_size = tensor.nbytes();
        let allocator = self
            .backend_allocator
            .as_ref()
            .ok_or_else(|| Error::msg("backend_allocator is none"))?;
        let padded_size = allocator.alloc_size(tensor)?;

        if padded_size <= original_size {
            return Ok(());
        }
        let storage = TensorStorage::new_cuda(Rc::new(*self), offset, padded_size);
        tensor.set_extra_storage(Some(storage));

        let stream = self.backend_ctx.unwrap().borrow().ensure_current_stream()?;
        let ptr = self.buffer.cu_deviceptr() + offset as u64;

        unsafe {
            memset_d8_async(ptr, 0, padded_size - original_size, stream.cu_stream()).map_err(
                |e| {
                    Error::msg(format!("memset_d8_async failed with error code {}", e))
                        .context("CudaBackendBuffer::init_tensor")
                },
            )?;
        }

        Ok(())
    }

    fn memset_tensor(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()> {
        self.backend_ctx.unwrap().set_device(self.backend_ctx.unwrap().current_device_id.unwrap());
        let storage = tensor.get_extra_storage()?;
        let offset = storage.offset();
        unsafe {
            let ptr = self.buffer.cu_deviceptr() as usize;
            memset_d8_async(
                (ptr + offset) as u64,
                value,
                size,
                self.backend_ctx.unwrap().current_stream.unwrap().cu_stream(),
            )
            .map_err(|e| {
                Error::msg(format!(
                    "cuda_core::memory::memset_d8_async failed with error code {}",
                    e
                ))
                .context("CudaBackendBuffer::memset_tensor")
            })?;

            self.backend_ctx.unwrap().current_stream.as_ref().unwrap().synchronize().map_err(
                |e| {
                    Error::msg(format!("cuStreamSynchronize failed with error code {}", e))
                        .context("CudaBackendBuffer::memset_tensor")
                },
            )?;
        }

        Ok(())
    }

    fn set_tensor(
        &self,
        tensor: crate::tensor::Tensor,
        data: &mut [u8],
        offset: usize,
        size: usize,
    ) -> crate::error::Result<()> {
        self.backend_ctx.unwrap().set_device(self.backend_ctx.unwrap().current_device_id.unwrap());
        let storage = tensor.get_extra_storage()?;
        let offset = storage.offset();

        unsafe {
            let ptr = self.buffer.cu_deviceptr() as usize;
            memcpy_htod_async(
                (ptr + offset) as u64,
                data.as_ptr() as *const std::ffi::c_void,
                size,
                self.backend_ctx.unwrap().current_stream.unwrap().cu_stream(),
            )
            .map_err(|e| {
                Error::msg(format!("memcpy_htod_async failed with error code {}", e))
                    .context("CudaBackendBuffer::set_tensor")
            })?;

            self.backend_ctx.unwrap().current_stream.as_ref().unwrap().synchronize().map_err(
                |e| {
                    Error::msg(format!("cuStreamSynchronize failed with error code {}", e))
                        .context("CudaBackendBuffer::memset_tensor")
                },
            )?;
        }

        Ok(())
    }

    fn get_tensor(
        &self,
        tensor: crate::tensor::Tensor,
        data: &mut [u8],
        offset: usize,
        size: usize,
    ) -> crate::error::Result<()> {
        self.backend_ctx.unwrap().set_device(self.backend_ctx.unwrap().current_device_id.unwrap());
        let storage = tensor.get_extra_storage()?;
        let offset = storage.offset();

        unsafe {
            let ptr = self.buffer.cu_deviceptr() as usize;
            memcpy_dtoh_async(
                data.as_mut_ptr(),
                (ptr + offset) as u64,
                size,
                self.backend_ctx.unwrap().current_stream.unwrap().cu_stream(),
            )
            .map_err(|e| {
                Error::msg(format!("memcpy_dtoh_async failed with error code {}", e))
                    .context("CudaBackendBuffer::get_tensor")
            })?;

            self.backend_ctx.unwrap().current_stream.as_ref().unwrap().synchronize().map_err(
                |e| {
                    Error::msg(format!("cuStreamSynchronize failed with error code {}", e))
                        .context("CudaBackendBuffer::memset_tensor")
                },
            )?;
        }

        Ok(())
    }

    fn copy_tensor(
        &self,
        src: crate::tensor::Tensor,
        dst: crate::tensor::Tensor,
    ) -> crate::error::Result<()> {
        let src_storage = src.get_extra_storage()?;
        let dst_storage = dst.get_extra_storage()?;

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

        let stream = src_backend_ctx.borrow().ensure_current_stream()?;
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

    fn get_usage(&self) -> Result<BackendBufferUsage> {
        Ok(self.usage.clone())
    }

    fn set_usage(&mut self, usage: BackendBufferUsage) -> Result<()> {
        self.usage = usage.clone();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
