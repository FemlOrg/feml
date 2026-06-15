use crate::compute_graph::ComputeGraph;
use crate::context::Context;
use crate::data_type::TensorOpType;
use crate::error::Result;
use crate::tensor::Tensor;
use std::any::Any;

pub enum BackendDeviceType {
    Cpu,
    Gpu,
    ACCEL,
}

#[derive(Default, Clone, Copy)]
pub struct BackendDeviceCaps {
    pub aysnc: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}

#[derive(Default, Clone)]
pub struct BackendDeviceProps {
    pub name: &'static str,
    pub description: String,
    pub memory_free: usize,
    pub memory_total: usize,
    pub device_type: BackendDeviceType,
    pub caps: BackendDeviceCaps,
}

#[derive(Default, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub memory: MemoryInfo,
    pub device_type: BackendDeviceType,
    pub caps: BackendCapabilities,
}

#[derive(Default, Clone, Copy)]
pub struct MemoryInfo {
    pub total: usize,
    pub free: usize,
}

#[derive(Default, Clone, Copy)]
pub struct BackendCapabilities {
    pub async_compute: bool,
    pub host_buffer: bool,
    pub events: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BackendBufferUsage {
    Any,
    Weights,
    Compute,
}

pub trait BackendBuffer {
    fn reset(&self) -> Result<()>;

    fn init_tensor(&self, tensor: Tensor, offset: usize) -> Result<()>;

    fn fill(&self, tensor: Tensor, value: u8, offset: usize, size: usize) -> Result<()>;

    fn write(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize) -> Result<()>;

    fn read(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize) -> Result<()>;

    fn copy(&self, src: Tensor, dst: Tensor) -> Result<()>;

    fn usage(&self) -> Result<BackendBufferUsage>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait Backend {
    fn name(&self) -> &str;

    fn synchronize(&self) -> Result<()>;

    fn graph_compute(&self, ctx: &Context, graph: &mut ComputeGraph) -> Result<()>;

    fn write_async(
        &self,
        tensor: Tensor,
        data: &mut [u8],
        offset: usize,
        size: usize,
    ) -> Result<()>;

    fn read_async(&self, tensor: Tensor, data: &mut [u8], offset: usize, size: usize)
    -> Result<()>;

    fn copy_async(&self, src: Tensor, dst: Tensor) -> Result<()>;

    fn create_buffer(
        &self,
        size: usize,
        usage: BackendBufferUsage,
    ) -> Result<Box<dyn BackendBuffer>>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait BackendDevice {
    fn info(&self) -> Result<DeviceInfo>;

    fn init_backend(&self) -> Result<Box<dyn Backend>>;

    fn supports_op(&self, op_type: TensorOpType) -> Result<bool>;

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

    fn probe_devices(&self) -> Result<()>;

    fn init_devices(&self) -> Result<()>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}
