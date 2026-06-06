use crate::compute_graph::ComputeGraph;
use crate::context::Context;
use crate::error::Result;
use crate::tensor::Tensor;
use std::any::Any;

pub enum BackendDeviceType {
    Cpu,
    Gpu,
    ACCEL,
}

pub struct BackendDeviceCaps {
    pub aysnc: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}

pub struct BackendDeviceProps {
    pub name: &'static str,
    pub description: String,
    pub memory_free: usize,
    pub memory_total: usize,
    pub device_type: BackendDeviceType,
    pub caps: BackendDeviceCaps,
}

// BackendBuffer = ggml_backend_buffer_type + ggml_backend_buffer
pub trait BackendBuffer {
    fn as_ptr(&self) -> Result<*mut u8>;

    fn device(&self) -> Result<Box<dyn BackendDevice>>;

    fn get_base(&self) -> Result<*mut u8>;

    fn clear(&self, value: u8) -> Result<()>;

    fn reset(&self) -> Result<()>;

    fn init_tensor(&self, tensor: Tensor, offset: usize) -> Result<()>;

    fn memset_tensor(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()>;

    fn set_tensor(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize)
        -> Result<()>;

    fn get_tensor(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize)
        -> Result<()>;

    fn copy_tensor(&self, src: Tensor, dst: Tensor) -> Result<()>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait BackendBufferAllocator {
    fn allocate_buffer(&self, size: usize) -> Result<Box<dyn BackendBuffer>>;

    fn alignment(&self) -> Result<usize> {
        Ok(std::mem::align_of::<usize>())
    }

    fn max_size(&self) -> Result<usize> {
        Ok(usize::MAX)
    }

    fn alloc_size(&self, tensor: Tensor) -> Result<usize>;

    fn is_host(&self) -> Result<bool>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait Backend {
    fn name(&self) -> &str;

    fn synchronize(&self) -> Result<()>;

    fn graph_compute(&self, ctx: &Context, graph: &mut ComputeGraph) -> Result<()>;

    fn memcpy_async(&self, dst: &mut [u8], src: &[u8], size: usize) -> Result<()>;

    fn set_tensor_async(
        &self,
        tensor: Tensor,
        data: *mut u8,
        offset: usize,
        size: usize,
    ) -> Result<()>;

    fn get_tensor_async(
        &self,
        tensor: Tensor,
        data: *mut u8,
        offset: usize,
        size: usize,
    ) -> Result<()>;

    fn copy_tensor_async(&self, src: Tensor, dst: Tensor) -> Result<()>;

    fn create_buffer_allocator(&self) -> Result<Box<dyn BackendBufferAllocator>>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait BackendDevice {
    fn name(&self) -> &str;

    fn memory(&self) -> (usize, usize);

    fn description(&self) -> &str;

    fn device_type(&self) -> BackendDeviceType;

    fn props(&self) -> BackendDeviceProps;

    fn init_backend(&self) -> Result<Box<dyn Backend>>;

    fn supports_op(&self, tensor: Tensor) -> Result<bool>;

    fn supports_buffer_allocator(
        &self,
        buffer_allocator: &dyn BackendBufferAllocator,
    ) -> Result<bool>;

    fn offload_op(&self, tensor: Tensor) -> Result<bool>;

    fn buffer_from_host_ptr(
        &self,
        ptr: &mut [u8],
        size: usize,
        max_tensor_size: usize,
    ) -> Result<Box<dyn BackendBuffer>>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait BackendRegister {
    fn name(&self) -> &str;

    fn device_count(&self) -> usize;

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}
