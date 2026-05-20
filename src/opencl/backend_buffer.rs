use crate::backend::BackendBuffer;
use crate::tensor::Tensor;
use ocl::Buffer;

pub struct OpenclBackendBuffer {
    buffers: Vec<ocl::Buffer<u8>>,
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

    fn set_tensor(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Ok(())
    }

    fn get_tensor(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Ok(())
    }

    fn copy_tensor(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Ok(())
    }
}
