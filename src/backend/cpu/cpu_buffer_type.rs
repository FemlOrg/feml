use crate::backend::backend::{FemlBackendBuffer, FemlBackendBufferType};
use crate::backend::backend_trait::FemlBackendBufferTypeInterface;
use crate::common::def::FEML_TENSOR_ALIGNMENT;
use crate::common::tensor::FemlTensor;

pub(crate) struct FemlBackendCpuBufferType;

impl FemlBackendBufferTypeInterface for FemlBackendCpuBufferType {
    fn get_name(&self, buffer_type: &FemlBackendBufferType) -> &'static str {
        "CPU"
    }

    // allocate a buffer of this type
    fn alloc_buffer(
        &self,
        buffer_type: &FemlBackendBufferType,
        size: usize,
    ) -> Option<FemlBackendBuffer> {
        None
    }

    // tensor alignment
    fn get_alignment(&self, buffer_type: &FemlBackendBufferType) -> usize {
        FEML_TENSOR_ALIGNMENT
    }

    // max buffer size that can be allocated (defaults to SIZE_MAX)
    fn get_max_size(&self, buffer_type: &FemlBackendBufferType) -> usize {
        0
    }

    // data size needed to allocate the tensor, including padding (defaults to feml_nbytes)
    fn get_alloc_size(
        &self,
        buffer_type: &FemlBackendBufferType,
        tensor: &mut FemlTensor,
    ) -> usize {
        0
    }

    // check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
    fn is_host(&self, buffer_type: &FemlBackendBufferType) -> bool {
        true
    }
}
