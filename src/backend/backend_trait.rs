use core::sync;
use std::sync::Arc;

use super::backend::*;
use crate::backend::cpu::compute_graph::FemlComputeGraph;
use crate::backend::cpu::cpu_register::BackendFunction;
use crate::common::tensor::FemlTensor;
use crate::types::FemlStatus;
use std::rc::Rc;

// use trait to implment backend dynamic polymorphism
pub trait FemlBackendBufferTypeInterface {
    fn get_name(&self, buffer_type: &FemlBackendBufferType) -> &'static str;

    // allocate a buffer of this type
    fn alloc_buffer(
        &self,
        buffer_type: &FemlBackendBufferType,
        size: usize,
    ) -> Option<FemlBackendBuffer>;

    // tensor alignment
    fn get_alignment(&self, buffer_type: &FemlBackendBufferType) -> usize;

    // max buffer size that can be allocated (defaults to SIZE_MAX)
    fn get_max_size(&self, buffer_type: &FemlBackendBufferType) -> usize;

    // data size needed to allocate the tensor, including padding (defaults to feml_nbytes)
    fn get_alloc_size(&self, buffer_type: &FemlBackendBufferType, tensor: &mut FemlTensor)
        -> usize;

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
    fn get_name(&self, backend: &FemlBackend) -> &'static str;

    fn free(&self, backend: &mut FemlBackend);

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

    fn graph_plan_create(&self, backend: &mut FemlBackend, compute_graph: &FemlComputeGraph);

    fn graph_plan_free(&self, backend: &FemlBackend, plan: *const u8);

    fn graph_plan_unpdate(
        &self,
        backend: &FemlBackend,
        plan: *const u8,
        compute_graph: &FemlComputeGraph,
    );

    fn graph_plan_compute(&self, backend: &FemlBackend, plan: *const u8) -> FemlStatus;

    fn graph_compute(&self, backend: &FemlBackend, compute_graph: &FemlComputeGraph);

    fn event_record(&self, backend: &FemlBackend, event: &FemlBackendEvent);

    fn event_wait(&self, backend: &FemlBackend, event: &FemlBackendEvent);
}

// TODO
fn feml_backend_buffer_copy_tensor(src: &FemlTensor, dst: &mut FemlTensor) {}

// TODO
// fn feml_backend_multi_buffer_alloc_buffer(buffers: &mut Vec<FemlBackendBuffer>, n_buffers: usize) -> FemlBackendBuffer {}

pub trait FemlBackendDeviceInterface {
    fn get_name(&self, device: &FemlBackendDevice) -> &'static str;

    fn get_description(&self, device: &FemlBackendDevice) -> String;

    fn get_memory(&self, device: &FemlBackendDevice) -> Result<(u64, u64), FemlStatus>;

    fn get_type(&self, device: &FemlBackendDevice) -> FemlBackendDeviceType;

    fn get_props(&self, device: &FemlBackendDevice, props: &mut FemlBackendDeviceProps);

    fn init_backend(&self, device: &FemlBackendDevice, params: &Vec<u8>);

    fn get_buffer_type(&self, device: &FemlBackendDevice) -> Option<FemlBackendBufferType>;

    fn get_host_buffer_type(&self, device: &FemlBackendDevice) -> Option<FemlBackendBufferType>;

    fn buffer_from_host_ptr(
        &self,
        device: &FemlBackendDevice,
        data: &Vec<u8>,
        max_tensor_size: usize,
    ) -> Option<FemlBackendBuffer>;

    fn support_op(&self, device: &FemlBackendDevice, op: &mut FemlTensor) -> bool;

    fn support_buft(&self, device: &FemlBackendDevice, buft: &FemlBackendBufferType) -> bool;

    fn offload_op(&self, device: &FemlBackendDevice, op: &mut FemlTensor) -> bool;

    fn event_new(&self, device: &FemlBackendDevice) -> Option<FemlBackendEvent>;

    fn event_free(&self, device: &FemlBackendDevice, event: &FemlBackendEvent);

    fn event_synchronize(&self, device: &FemlBackendDevice, event: &FemlBackendEvent);
}

pub trait FemlBackendRegInterface: Send + Sync {
    fn get_name(&self, reg: &FemlBackendReg) -> &'static str;

    fn get_device_count(&self, reg: &FemlBackendReg) -> usize;

    fn get_device(&self, reg: &Arc<FemlBackendReg>, index: usize) -> Option<FemlBackendDevice>;

    fn get_proc_address(&self, reg: &FemlBackendReg, name: &str) -> BackendFunction;
}
