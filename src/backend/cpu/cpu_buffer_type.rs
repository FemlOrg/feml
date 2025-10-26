use std::sync::Arc;

use crate::backend::backend::{FemlBackendBuffer, FemlBackendBufferType, feml_backend_buffer_init};
use crate::backend::backend_trait::{FemlBackendBufferInterface, FemlBackendBufferTypeInterface};
use crate::backend::cpu::cpu_buffer_backend::FemlBackendCpuBufferImpl;
use crate::common::def::FEML_TENSOR_ALIGNMENT;
use crate::common::tensor::FemlTensor;
use crate::feml_abort;
use crate::feml_impl::feml_aligned_malloc;

fn feml_backend_cpu_buffer_type_get_name() -> &'static str {
    "CPU"
}

fn feml_backend_cpu_buffer_type_alloc_buffer(
    buffer_type: &Arc<FemlBackendBufferType>,
    size: usize,
) -> FemlBackendBuffer {
    let data = feml_aligned_malloc(size);

    let mut interface: Option<Box<dyn FemlBackendBufferInterface>> =
        Some(Box::new(FemlBackendCpuBufferImpl));

    feml_backend_buffer_init(buffer_type.clone(), &mut interface, Some(Box::new(data)), size)
}

fn feml_backend_cpu_buffer_type_get_alignment() -> usize {
    FEML_TENSOR_ALIGNMENT
}

fn feml_backend_cpu_buffer_type_is_host() -> bool {
    true
}
pub(crate) struct FemlBackendCpuBufferTypeImpl;

impl FemlBackendBufferTypeInterface for FemlBackendCpuBufferTypeImpl {
    fn get_name(&self, _buffer_type: &FemlBackendBufferType) -> &'static str {
        feml_backend_cpu_buffer_type_get_name()
    }

    // allocate a buffer of this type
    fn alloc_buffer(
        &self,
        buffer_type: &Arc<FemlBackendBufferType>,
        size: usize,
    ) -> Option<FemlBackendBuffer> {
        Some(feml_backend_cpu_buffer_type_alloc_buffer(buffer_type, size))
    }

    // tensor alignment
    fn get_alignment(&self, _buffer_type: &FemlBackendBufferType) -> usize {
        feml_backend_cpu_buffer_type_get_alignment()
    }

    // max buffer size that can be allocated (defaults to SIZE_MAX)
    fn get_max_size(&self, _buffer_type: &FemlBackendBufferType) -> usize {
        feml_abort!("FemlBackendCpuBufferTypeImpl not implement get_max_size");
    }

    // data size needed to allocate the tensor, including padding (defaults to feml_nbytes)
    fn get_alloc_size(
        &self,
        _buffer_type: &FemlBackendBufferType,
        _tensor: &mut FemlTensor,
    ) -> usize {
        feml_abort!("FemlBackendCpuBufferTypeImpl not implement get_alloc_size");
    }

    // check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
    fn is_host(&self, _buffer_type: &FemlBackendBufferType) -> bool {
        feml_backend_cpu_buffer_type_is_host()
    }
}
