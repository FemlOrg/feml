use crate::backend::BackendBuffer;
use crate::data_type::get_type_size;
use crate::tensor::Tensor;

pub struct CpuBackendBuffer {
    buffers: Vec<u8>,
}

impl BackendBuffer for CpuBackendBuffer {
    fn init_tensor(&self, tensor: Tensor, offset: usize) -> Result<()> {
        Ok(())
    }

    fn memset_tensor(
        &mut self,
        _tensor: Tensor,
        _value: u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        let size = self.buffers.len();
        if _offset + _size > size {
            return Err("offset + size > buffer size".to_string());
        }
        self.buffers[_offset.._offset + _size].fill(_value);
        Ok(())
    }

    fn set_tensor(
        &mut self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        if _offset + _size > self.buffers.len() {
            return Err("offset + size > buffer size".to_string());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(_data, self.buffers.as_mut_ptr().add(_offset), _size);
        }
        Ok(())
    }

    fn get_tensor(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        if _offset + _size > self.buffers.len() {
            return Err("offset + size > buffer size".to_string());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.buffers.as_ptr().add(_offset), _data, _size);
        }
        Ok(())
    }

    fn copy_tensor(&mut self, _src: Tensor, _dst: Tensor) -> Result<()> {
        let src_inner = _src.borrow().storage.as_ref().unwrap();
        let dst_inner = _dst.borrow().storage.as_ref().unwrap();
        unsafe {
            let Some(TensorStorage::Cpu { offset: src_offset, .. }) = src_inner else {
                return Err("source tensor is not a CPU tensor".to_string());
            };
            let Some(TensorStorage::Cpu { offset: dst_offset, .. }) = dst_inner else {
                return Err("destination tensor is not a CPU tensor".to_string());
            };
            std::ptr::copy_nonoverlapping(
                self.buffers.as_ptr().add(dst_offset),
                self.buffers.as_ptr().add(src_offset),
                _src.borrow().length(),
            );
        }
        Ok(())
    }
}
