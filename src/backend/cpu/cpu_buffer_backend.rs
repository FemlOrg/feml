use crate::backend::backend::{FemlBackendBuffer, FemlBackendBufferType};
use crate::backend::backend_trait::FemlBackendBufferInterface;
use crate::common::tensor::FemlTensor;
use crate::types::FemlStatus;

pub(crate) struct FemlBackendCpuBuffer;

impl FemlBackendBufferInterface for FemlBackendCpuBuffer {
    // free the buffer
    fn free_buffer(&self, buffer: &FemlBackendBuffer) {}

    // base address of the buffer
    fn get_base(&self, buffer: &FemlBackendBuffer) {}

    // initialize a tensor in the buffer (eg. add tensor extras)
    fn init_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor) -> FemlStatus {
        todo!("init_tensor");
    }

    // tensor data access
    fn memset_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor) {
        todo!("memset_tensor");
    }

    fn set_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor) {
        todo!("set_tensor");
    }

    fn get_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor) {
        todo!("get_tensor");
    }

    // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    fn cpy_tensor(
        &self,
        buffer: &FemlBackendBuffer,
        src: &FemlTensor,
        dst: &mut FemlTensor,
    ) -> bool {
        todo!("cpy_tensor");
    }

    // clear the entire buffer
    fn clear(&self, buffer: &FemlBackendBuffer, value: u8) {
        todo!("clear");
    }

    // (optional) reset any internal state due to tensor initialization, such as tensor extras
    fn reset(&self, buffer: &FemlBackendBuffer) {
        todo!("reset");
    }
}
