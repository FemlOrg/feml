use crate::backend::{BackendBuffer, BackendBufferUsage};
use crate::error::{Error, Result};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;
use std::any::Any;
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

#[derive(Clone)]
pub struct CpuBackendBuffer {
    buffers: Rc<RefCell<Vec<u8>>>,
    usage: BackendBufferUsage,
}

impl CpuBackendBuffer {
    pub(super) fn new(size: usize, usage: BackendBufferUsage) -> Self {
        Self { buffers: Rc::new(RefCell::new(vec![0; size])), usage }
    }

    fn len(&self) -> usize {
        self.buffers.borrow().len()
    }

    fn checked_range(&self, start: usize, size: usize) -> Result<Range<usize>> {
        let end = start.checked_add(size).ok_or_else(|| Error::msg("offset + size overflow"))?;

        if end > self.len() {
            return Err(Error::msg("offset + size > buffer size"));
        }

        Ok(start..end)
    }

    fn tensor_range(&self, tensor: &Tensor, offset: usize, size: usize) -> Result<Range<usize>> {
        let storage = tensor.storage()?;

        if !matches!(*storage, TensorStorage::Cpu { .. }) {
            return Err(Error::msg("storage is not CPU type"));
        }

        let start = storage
            .offset()
            .checked_add(tensor.view_offset())
            .and_then(|offset_with_view| offset_with_view.checked_add(offset))
            .ok_or_else(|| Error::msg("offset + size overflow"))?;

        self.checked_range(start, size)
    }
}

impl BackendBuffer for CpuBackendBuffer {
    fn reset(&self) -> Result<()> {
        self.buffers.borrow_mut().fill(0);
        Ok(())
    }

    fn init_tensor(&self, mut tensor: Tensor, offset: usize) -> Result<()> {
        let view_tensor_opt = tensor.borrow().view_tensor.clone();
        match view_tensor_opt {
            Some(view_tensor) => {
                let view_storage = view_tensor.storage()?.clone();
                tensor.set_storage(Some(view_storage))?;
            }
            None => {
                self.checked_range(offset, tensor.nbytes())?;
                let storage =
                    TensorStorage::new_cpu(Rc::new(self.clone()), offset, tensor.nbytes());
                tensor.set_storage(Some(storage))?;
            }
        }

        Ok(())
    }

    fn fill(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()> {
        let range = self.tensor_range(&tensor, offset, size)?;
        self.buffers.borrow_mut()[range].fill(value);
        Ok(())
    }

    fn write(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize) -> Result<()> {
        if size > data.len() {
            return Err(Error::msg("size > data length"));
        }

        let range = self.tensor_range(&tensor, offset, size)?;
        self.buffers.borrow_mut()[range].copy_from_slice(&data[..size]);
        Ok(())
    }

    fn read(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize) -> Result<()> {
        if size > data.len() {
            return Err(Error::msg("size > data length"));
        }

        let range = self.tensor_range(&tensor, offset, size)?;
        data[..size].copy_from_slice(&self.buffers.borrow()[range]);
        Ok(())
    }

    fn copy(&self, src: Tensor, dst: Tensor) -> Result<()> {
        let src_storage = src.storage()?;
        let dst_storage = dst.storage()?;

        let src_buffer =
            src_storage.as_cpu().ok_or_else(|| Error::msg("src tensor storage is not CPU"))?;
        let dst_buffer =
            dst_storage.as_cpu().ok_or_else(|| Error::msg("dst tensor storage is not CPU"))?;

        let size = src.nbytes();
        if size > dst.nbytes() {
            return Err(Error::msg("source tensor is larger than destination tensor"));
        }

        let src_start = src_storage
            .offset()
            .checked_add(src.view_offset())
            .ok_or_else(|| Error::msg("source offset overflow"))?;
        let dst_start = dst_storage
            .offset()
            .checked_add(dst.view_offset())
            .ok_or_else(|| Error::msg("destination offset overflow"))?;
        let src_range = src_buffer.checked_range(src_start, size)?;
        let dst_range = dst_buffer.checked_range(dst_start, size)?;

        if Rc::ptr_eq(&src_buffer.buffers, &dst_buffer.buffers) {
            src_buffer.buffers.borrow_mut().copy_within(src_range, dst_start);
        } else {
            let src_buffers = src_buffer.buffers.borrow();
            let mut dst_buffers = dst_buffer.buffers.borrow_mut();
            dst_buffers[dst_range].copy_from_slice(&src_buffers[src_range]);
        }

        Ok(())
    }

    fn usage(&self) -> Result<BackendBufferUsage> {
        Ok(self.usage)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
