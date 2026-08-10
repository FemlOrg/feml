use super::backend_context::CpuBackendContext;
use super::backend_device::CpuBackendDevice;
use super::backend_register::CpuBackendRegister;
use crate::backend::Backend;
use crate::compute_graph::ComputeGraph;
use crate::error::{Error, ErrorKind, Result};
use crate::tensor::Tensor;

pub struct CpuBackend {
    device: CpuBackendDevice,
    context: CpuBackendContext,
}

impl Backend for CpuBackend {
    type Device = CpuBackendDevice;

    fn name(&self) -> &str {
        "cpu"
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn graph_compute(&self, _graph: &mut ComputeGraph) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "cpu", op: "graph_compute" }))
    }

    fn memcpy_async(&self, _dst: &mut [u8], _src: &[u8], _size: usize) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "cpu", op: "memcpy_async" }))
    }

    fn set_tensor_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "cpu", op: "set_tensor_async" }))
    }

    fn get_tensor_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "cpu", op: "get_tensor_async" }))
    }

    fn copy_tensor_async(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "cpu", op: "copy_tensor_async" }))
    }
}

impl CpuBackend {
    pub fn init() -> Result<Self> {
        let reg = CpuBackendRegister::init();
        let device = reg.cpu_device(0)?;
        let context = CpuBackendContext::new();

        Ok(Self { device, context })
    }
}
