use super::backend_device::OpenclBackendDevice;
use crate::backend::BackendBuffer;
use crate::error::Error;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use ocl::Buffer;

pub(super) struct OpenclBuffer {
    buffer: ocl::Buffer<u8>,
    size: usize,
}

pub struct OpenclBackendBuffer {
    backend_device: Option<OpenclBackendDevice>,
    buffers: Vec<OpenclBuffer>,
}

impl BackendBuffer for OpenclBackendBuffer {
    fn init_tensor(&self, tensor: Tensor, offset: usize) -> Result<()> {
        match tensor.borrow().view_tensor.clone() {
            Some(view_tensor) => {
                let extra_storage = view_tensor
                    .borrow()
                    .extra_storage
                    .as_ref()
                    .ok_or_else(|| Error::msg("view extra_storage is None"))?
                    .clone();

                tensor.borrow_mut().extra_storage = Some(extra_storage);
            }
            None => {
                tensor.borrow_mut().extra_storage = Some(TensorStorage::OpenCL {
                    buffer: Rc::new(self),
                    ocl_buffer_idx: 0,
                    offset: offset,
                    acutal_size: tensor.nbytes(),
                });
            }
        }
        Ok(())
    }

    fn memset_tensor(
        &self,
        _tensor: Tensor,
        _value: u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Ok(())
    }

    fn set_tensor(&self, tensor: Tensor, data: [u8], offset: usize, size: usize) -> Result<()> {
        // TODO : support quant type
        self.backend_device.unwrap().init()?;

        let mut backend_ctx = self.backend_device.unwrap().backend_ctx;
        let cl_context = backend_ctx.unwrap().context;
        let cl_queue = backend_ctx.unwrap().queue;

        let storage = tensor
            .borrow()
            .extra_storage
            .as_ref()
            .ok_or_else(|| Error::msg("extra_storage is none"))?;

        if let TensorStorage::OpenCL { ocl_buffer_idx, .. } = storage {
            let buffer = &self.buffers[*ocl_buffer_idx].buffer;
            unsafe {
                ocl::core::enqueue_write_buffer(
                    &cl_queue,
                    buffer,
                    true,
                    offset,
                    &data,
                    None::<ocl::core::Event>,
                    None::<ocl::core::Event>,
                )?;
            }
        } else {
            return Err(Error::msg("storage is not OpenCL type"));
        }
        Ok(())
    }

    fn get_tensor(&self, tensor: Tensor, mut data: [u8], offset: usize, size: usize) -> Result<()> {
        // TODO: before read buffer, add sync_with_other_backends
        // TODO : support quant type
        self.backend_device.unwrap().init()?;

        let mut backend_ctx = self.backend_device.unwrap().backend_ctx;
        let cl_context = backend_ctx.unwrap().context;
        let cl_queue = backend_ctx.unwrap().queue;

        let storage = tensor
            .borrow()
            .extra_storage
            .as_ref()
            .ok_or_else(|| Error::msg("extra_storage is none"))?;

        if let TensorStorage::OpenCL { ocl_buffer_idx, .. } = storage {
            let buffer = &self.buffers[*ocl_buffer_idx].buffer;
            unsafe {
                ocl::core::enqueue_read_buffer(
                    &cl_queue,
                    buffer,
                    true,
                    offset,
                    &mut data,
                    None::<ocl::core::Event>,
                    None::<ocl::core::Event>,
                )?;
            }
        } else {
            return Err(Error::msg("storage is not OpenCL type"));
        }

        Ok(())
    }

    fn copy_tensor(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Ok(())
    }

    fn as_ptr(&self) -> *mut u8 {
        todo!()
    }

    fn device(&self) -> Box<dyn crate::backend::BackendDevice> {
        todo!()
    }

    fn get_base(&self) -> *mut u8 {
        todo!()
    }

    fn clear(&self, value: u8) -> Result<()> {
        self.backend_device.unwrap().init()?;

        let mut backend_ctx = self.backend_device.unwrap().backend_ctx;
        let cl_context = backend_ctx.unwrap().context;
        let cl_queue = backend_ctx.unwrap().queue;

        for buf in self.buffers.iter() {
            unsafe {
                ocl::core::enqueue_fill_buffer(
                    &cl_queue,
                    buf.buffer,
                    value,
                    offset,
                    buf.size,
                    None::<ocl::core::Event>,
                    None::<ocl::core::Event>,
                    None,
                )?;
            }
        }

        cl_queue.finish();
        Ok(())
    }

    fn reset(&self) -> Result<()> {
        todo!("add context reset")
    }
}
