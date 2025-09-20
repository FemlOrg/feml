use super::backend::*;
use crate::common::tensor::FemlTensor;
use crate::types::FemlStatus;

// use trait to implment backend dynamic polymorphism
pub trait FemlBackendBufferTypeInterface {
    fn get_name(&self, buffer_type: &FemlBackendBufferType) -> *const str;

    // allocate a buffer of this type
    fn alloc_buffer(&self, buffer_type: &FemlBackendBufferType, size: usize) -> FemlBackendBuffer;

    // tensor alignment
    fn get_alignment(&self, buffer_type: &FemlBackendBufferType) -> usize;

    // max buffer size that can be allocated (defaults to SIZE_MAX)
    fn get_max_size(&self, buffer_type: &FemlBackendBufferType) -> usize;

    // data size needed to allocate the tensor, including padding (defaults to feml_nbytes)
    fn get_alloc_size(&self, buffer_type: &FemlBackendBufferType, tensor: &mut FemlTensor);

    // check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
    fn is_host(&self, buffer_type: &FemlBackendBufferType) -> bool;
}

pub trait FemlBackendBufferInterface {
    // free the buffer
    fn free_buffer(&self, buffer: &FemlBackendBuffer);

    // base address of the buffer
    fn get_base(&self, buffer: &FemlBackendBuffer);

    // initialize a tensor in the buffer (eg. add tensor extras)
    fn init_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor) -> FemlStatus;

    // tensor data access
    fn memset_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor);

    fn set_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor);

    fn get_tensor(&self, buffer: &FemlBackendBuffer, tensor: &mut FemlTensor);

    // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
    fn cpy_tensor(
        &self,
        buffer: &FemlBackendBuffer,
        src: &FemlTensor,
        dst: &mut FemlTensor,
    ) -> bool;

    // clear the entire buffer
    fn clear(&self, buffer: &FemlBackendBuffer, value: u8);

    // (optional) reset any internal state due to tensor initialization, such as tensor extras
    fn reset(&self, buffer: &FemlBackendBuffer);
}

pub trait FemlBackendInterface {
    fn get_name(&self, backend: &FemlBackend) -> *const str;

    fn free(&self, backend: &FemlBackend);

    // asynchronous tensor data access
    fn set_tensor_async(
        &self,
        backend: &FemlBackend,
        tensor: &mut FemlTensor,
        data: *const u8,
        offset: usize,
        size: usize,
    );

    fn get_tensor_async(
        &self,
        backend: &FemlBackend,
        tensor: &mut FemlTensor,
        data: *const u8,
        offset: usize,
        size: usize,
    );

    fn cpy_tensor_async(
        &self,
        bakend_src: &FemlBackend,
        backend_dst: &FemlBackend,
        src: &FemlTensor,
        dst: &mut FemlTensor,
    ) -> bool;

    // complete all pending operations (required if the backend supports async operations)
    fn synchronize(&self, backend: &FemlBackend);

    // TODO: Add compute graph
    // fn graph_plan_create(backend: &FemlBackend, compute_graph: &FemlComputeGraph);
    // fn graph_plan_free(backend: &FemlBackend, plan: *const u8);
    // fn graph_plan_unpdate(backend: &FemlBackend, plan: *const u8, compute_graph:& FemlComputeGraph);
    // fn graph_plan_compute(backend: &FemlBackend, plan: *const u8) -> FemlStatus;
    // fn graph_compute(backend: &FemlBackend, compute_graph: &FemlComputeGraph);

    fn event_record(&self, backend: &FemlBackend, event: &FemlBackendEvent);
    fn event_wait(&self, backend: &FemlBackend, event: &FemlBackendEvent);
}

// TODO
impl FemlBackendBufferType {
    // fn feml_backend_buffer_init(&self, interface: Box<dyn FemlBackendBufferInterface>, context: *mut u8, size: usize) -> FemlBackendBufferType{}

    // fn feml_backend_buffer_is_multi_buffer(&self) -> bool {}

    // fn feml_backend_multi_buffer_set_usage(&self, usage: &FemlBackendBufferUsage);
}

// TODO
fn feml_backend_buffer_copy_tensor(src: &FemlTensor, dst: &mut FemlTensor) {}

// TODO
// fn feml_backend_multi_buffer_alloc_buffer(buffers: &mut Vec<FemlBackendBuffer>, n_buffers: usize) -> FemlBackendBuffer {}
