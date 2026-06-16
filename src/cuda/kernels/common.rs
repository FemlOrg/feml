use crate::error::Result;
use crate::tensor::Tensor;

pub(super) fn get_cuda_tensor_ptr_and_size(tensor: Tensor) -> Result<(u64, usize)> {
    let storage = tensor.storage()?;
    let buffer = storage.as_cuda().unwrap();
    let offset = storage.offset();
    let size = storage.size();
    let ptr = buffer.buffer.cu_deviceptr() + offset as u64;
    Ok((ptr, size))
}
