use super::backend_context::OpenclBackendContext;
use crate::backend::{BackendBuffer, BackendBufferUsage};
use crate::error::{Error, Result};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct OpenclBackendBuffer {
    pub(super) backend_ctx: Option<Rc<RefCell<OpenclBackendContext>>>,
    pub(crate) buffer: ocl::Buffer<u8>,
    pub(crate) usage: BackendBufferUsage,
    pub(crate) size: usize,
}

impl OpenclBackendBuffer {
    pub(super) fn new(
        backend_ctx: Rc<RefCell<OpenclBackendContext>>,
        buffer: ocl::Buffer<u8>,
        usage: BackendBufferUsage,
        size: usize,
    ) -> Self {
        OpenclBackendBuffer { backend_ctx: Some(backend_ctx), buffer, usage, size }
    }
}

impl BackendBuffer for OpenclBackendBuffer {
    fn init_tensor(&self, mut tensor: Tensor, offset: usize) -> Result<()> {
        let view_tensor_opt = tensor.borrow().view_tensor.clone();
        match view_tensor_opt {
            Some(view_tensor) => {
                let view_extra = view_tensor.get_storage()?.clone();
                tensor.set_storage(Some(view_extra))?;
            }
            None => {
                let storage =
                    TensorStorage::new_opencl(Rc::new(self.clone()), offset, tensor.nbytes());
                tensor.set_storage(Some(storage))?;
            }
        }
        Ok(())
    }

    fn fill(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        let tensor_ref = tensor.borrow();
        let storage = tensor_ref.storage.as_ref().ok_or_else(|| Error::msg("storage is none"))?;

        if !matches!(storage, TensorStorage::Opencl { .. }) {
            return Err(Error::msg("storage is not OpenCL type"));
        }

        ocl::core::enqueue_fill_buffer(
            cl_queue,
            &self.buffer,
            value,
            offset,
            size,
            None::<ocl::core::Event>,
            None::<()>,
            None,
        )
        .map_err(|e| Error::msg(format!("OpenCL fill buffer failed: {}", e)))?;
        Ok(())
    }

    fn write(&self, tensor: Tensor, data: &mut [u8], offset: usize, _size: usize) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        let tensor_ref = tensor.borrow();
        let storage = tensor_ref.storage.as_ref().ok_or_else(|| Error::msg("storage is none"))?;

        if !matches!(storage, TensorStorage::Opencl { .. }) {
            return Err(Error::msg("storage is not OpenCL type"));
        }

        unsafe {
            ocl::core::enqueue_write_buffer(
                cl_queue,
                &self.buffer,
                true,
                offset,
                data,
                None::<ocl::core::Event>,
                None::<()>,
            )
            .map_err(|e| Error::msg(format!("OpenCL write buffer failed: {}", e)))?;
        }
        Ok(())
    }

    fn read(&self, tensor: Tensor, data: &mut [u8], offset: usize, _size: usize) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        let tensor_ref = tensor.borrow();
        let storage = tensor_ref.storage.as_ref().ok_or_else(|| Error::msg("storage is none"))?;

        if !matches!(storage, TensorStorage::Opencl { .. }) {
            return Err(Error::msg("storage is not OpenCL type"));
        }

        unsafe {
            ocl::core::enqueue_read_buffer(
                cl_queue,
                &self.buffer,
                true,
                offset,
                data,
                None::<ocl::core::Event>,
                None::<()>,
            )
            .map_err(|e| Error::msg(format!("OpenCL read buffer failed: {}", e)))?;
        }

        Ok(())
    }

    fn copy(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Err(Error::msg(format!("opencl: copy_tensor is not implemented yet"))
            .context("in OpenclBackendBuffer::copy_tensor"))
    }

    fn reset(&self) -> Result<()> {
        todo!("add context reset")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_usage(&self) -> Result<BackendBufferUsage> {
        Ok(self.usage)
    }
}
