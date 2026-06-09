use super::backend_context::OpenclBackendContext;
use crate::backend::{BackendBuffer, BackendBufferUsage};
use crate::error::{Error, Result};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct OpenclBackendBuffer {
    pub(crate) backend_ctx: Option<Rc<RefCell<OpenclBackendContext>>>,
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
        tensor.borrow_mut().view_tensor.map_or_else(
            || {
                let storage = TensorStorage::new_opencl(Rc::new(*self), offset, tensor.nbytes());
                tensor.set_extra_storage(Some(storage));
            },
            |view_tensor| {
                let view_extra = view_tensor.get_extra_storage()?.clone();
                tensor.set_extra_storage(Some(view_extra));
            },
        );
        Ok(())
    }

    fn memset_tensor(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        let tensor_ref = tensor.borrow();
        let storage =
            tensor_ref.extra_storage.as_ref().ok_or_else(|| Error::msg("extra_storage is none"))?;

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

    fn set_tensor(
        &self,
        tensor: Tensor,
        data: &mut [u8],
        offset: usize,
        _size: usize,
    ) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        let tensor_ref = tensor.borrow();
        let storage =
            tensor_ref.extra_storage.as_ref().ok_or_else(|| Error::msg("extra_storage is none"))?;

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

    fn get_tensor(
        &self,
        tensor: Tensor,
        data: &mut [u8],
        offset: usize,
        _size: usize,
    ) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        let tensor_ref = tensor.borrow();
        let storage =
            tensor_ref.extra_storage.as_ref().ok_or_else(|| Error::msg("extra_storage is none"))?;

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

    fn copy_tensor(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Err(Error::msg(format!("opencl: copy_tensor is not implemented yet"))
            .context("in OpenclBackendBuffer::copy_tensor"))
    }

    fn as_ptr(&self) -> Result<*mut u8> {
        Ok(self.buffer.as_core().as_ptr() as *mut u8)
    }

    fn device(&self) -> Result<Box<dyn crate::backend::BackendDevice>> {
        Err(Error::msg(format!("opencl: device is not implemented yet"))
            .context("in OpenclBackendBuffer::device"))
    }

    fn get_base(&self) -> Result<*mut u8> {
        Err(Error::msg(format!("opencl: get_base is not implemented yet"))
            .context("in OpenclBackendBuffer::get_base"))
    }

    fn clear(&self, value: u8) -> Result<()> {
        let ctx = self.backend_ctx.as_ref().unwrap().borrow();
        let cl_queue = &ctx.queue;

        ocl::core::enqueue_fill_buffer(
            cl_queue,
            &self.buffer,
            value,
            0,
            self.size,
            None::<ocl::core::Event>,
            None::<()>,
            None,
        )
        .map_err(|e| Error::msg(format!("OpenCL fill buffer failed: {}", e)))?;

        cl_queue.finish()?;
        Ok(())
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

    fn set_usage(&mut self, usage: BackendBufferUsage) -> Result<()> {
        self.usage = usage;
        Ok(())
    }
}
