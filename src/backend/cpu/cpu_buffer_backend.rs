use crate::backend::api::feml_backend_buffer_is_host;
use crate::backend::backend::FemlBackendBuffer;
use crate::backend::backend_trait::FemlBackendBufferInterface;
use crate::common::def::FEML_TENSOR_ALIGNMENT;
use crate::common::tensor::{FemlTensor, feml_nbytes};
use crate::feml_impl::feml_aligned_free;
use crate::types::FemlStatus;
use crate::{feml_abort, feml_pad};

pub(crate) struct FemlBackendCpuBufferImpl;

impl FemlBackendBufferInterface for FemlBackendCpuBufferImpl {
    // free the buffer
    fn free_buffer(&self, buffer: &mut FemlBackendBuffer) {
        feml_backend_cpu_buffer_free_buffer(buffer);
    }

    // base address of the buffer
    fn get_base(&self, buffer: &mut FemlBackendBuffer) -> *mut u8 {
        feml_backend_cpu_buffer_get_base(buffer)
    }

    // initialize a tensor in the buffer (eg. add tensor extras)
    fn init_tensor(&self, _buffer: &FemlBackendBuffer, _tensor: &mut FemlTensor) -> FemlStatus {
        feml_abort!("FemlBackendCpuBufferImpl not implement init_tensor trait");
    }

    // tensor data access
    fn memset_tensor(
        &self,
        _buffer: &FemlBackendBuffer,
        tensor: &mut FemlTensor,
        value: u8,
        offset: usize,
        size: usize,
    ) {
        feml_backend_cpu_buffer_memset_tensor(_buffer, tensor, value, offset, size);
    }

    fn set_tensor(
        &self,
        buffer: &FemlBackendBuffer,
        tensor: &mut FemlTensor,
        data: *const u8,
        offset: usize,
        size: usize,
    ) {
        feml_backend_cpu_buffer_set_tensor(buffer, tensor, data, offset, size);
    }

    fn get_tensor(
        &self,
        buffer: &FemlBackendBuffer,
        tensor: &mut FemlTensor,
        data: *mut u8,
        offset: usize,
        size: usize,
    ) {
        feml_backend_cpu_buffer_get_tensor(buffer, tensor, data, offset, size);
    }

    // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    fn cpy_tensor(
        &self,
        buffer: &FemlBackendBuffer,
        src: &FemlTensor,
        dst: &mut FemlTensor,
    ) -> bool {
        feml_backend_cpu_buffer_cpy_tensor(buffer, src, dst)
    }

    // clear the entire buffer
    fn clear(&self, buffer: &mut FemlBackendBuffer, value: u8) {
        feml_backend_cpu_buffer_clear(buffer, value);
    }

    // (optional) reset any internal state due to tensor initialization, such as tensor extras
    fn reset(&self, _buffer: &FemlBackendBuffer) {
        feml_abort!("FemlBackendCpuBufferImpl not implement reset trait");
    }
}

fn feml_backend_cpu_buffer_free_buffer(buffer: &mut FemlBackendBuffer) {
    let ctx = buffer.get_context::<*mut u8>().unwrap();
    feml_aligned_free(*ctx, buffer.size);
}

fn feml_backend_cpu_buffer_get_base(buffer: &mut FemlBackendBuffer) -> *mut u8 {
    let ctx = buffer.get_context::<Vec<u8>>().unwrap();
    let mut ctx_align = ctx.as_ptr() as usize;

    if ctx_align % FEML_TENSOR_ALIGNMENT != 0 {
        ctx_align = feml_pad!(ctx_align, FEML_TENSOR_ALIGNMENT);
    }

    ctx_align as *mut u8
}

fn feml_backend_cpu_buffer_memset_tensor(
    _buffer: &FemlBackendBuffer,
    tensor: &mut FemlTensor,
    value: u8,
    offset: usize,
    size: usize,
) {
    unsafe {
        std::ptr::write_bytes(tensor.data.add(offset), value, size);
    }
}

fn feml_backend_cpu_buffer_set_tensor(
    _buffer: &FemlBackendBuffer,
    tensor: &mut FemlTensor,
    data: *const u8,
    offset: usize,
    size: usize,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(data, tensor.data.add(offset), size);
    }
}

fn feml_backend_cpu_buffer_get_tensor(
    _buffer: &FemlBackendBuffer,
    tensor: &mut FemlTensor,
    data: *mut u8,
    offset: usize,
    size: usize,
) {
    unsafe {
        std::ptr::copy_nonoverlapping(tensor.data.add(offset), data, size);
    }
}
fn feml_backend_cpu_buffer_cpy_tensor(
    buffer: &FemlBackendBuffer,
    src_tensor: &FemlTensor,
    dst: &mut FemlTensor,
) -> bool {
    if feml_backend_buffer_is_host(buffer) {
        unsafe {
            std::ptr::copy_nonoverlapping(src_tensor.data, dst.data, feml_nbytes(src_tensor));
        }
    }
    false
}

fn feml_backend_cpu_buffer_clear(buffer: &mut FemlBackendBuffer, value: u8) {
    unsafe {
        std::ptr::write_bytes(*buffer.get_context::<*mut u8>().unwrap(), value, buffer.size);
    }
}
