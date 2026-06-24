use super::backend::CudaBackend;
use super::backend_context::CudaBackendContext;
use crate::backend::{Backend, BackendBuffer, BackendDevice, DeviceInfo};
use crate::data_type::TensorOpType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use cuda_core::CudaContext;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Default)]
pub(super) struct CudaDeviceInfo {
    pub(super) device_id: i32,
    pub(super) name: String,
    pub(super) max_threads_per_block: i32, // max number of threads per block
    pub(super) max_threads_dim: [i32; 3],  // max size of each dimension of a block
    pub(super) max_grid_size: [i32; 3],    // max size of each dimension of a block
    pub(super) shared_mem_per_block: i32,  // max. shared memory per block in bytes
    pub(super) total_constant_mem: i32,    // constant memory available on device in bytes
    pub(super) regs_per_block: i32,        // 32-bit  registers available per block
    pub(super) clock_rate: i32,            // clock frequency in kilohertz
    pub(super) warp_size: i32,             // Number of threads in a dispatch
}

#[derive(Clone)]
pub struct CudaBackendDevice {
    pub(super) info: CudaDeviceInfo,
    pub(super) context: Arc<CudaContext>,
    pub(super) backend_ctx: Option<Rc<RefCell<CudaBackendContext>>>,
}

impl BackendDevice for CudaBackendDevice {
    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        let ctx = self.backend_ctx.clone().ok_or_else(|| Error::msg("backend_ctx is none"))?;
        ctx.borrow_mut().set_device(self.info.device_id as u32)?;
        Ok(Box::new(CudaBackend { backend_ctx: ctx }))
    }

    fn supports_op(&self, op_type: TensorOpType) -> Result<bool> {
        match op_type {
            TensorOpType::TensorOpMul => Ok(true),
            _ => Ok(false),
        }
    }

    fn offload_op(&self, _tensor: Tensor) -> Result<bool> {
        todo!()
    }

    fn buffer_from_host_ptr(
        &self,
        _ptr: &mut [u8],
        _size: usize,
        _max_tensor_size: usize,
    ) -> Result<Box<dyn BackendBuffer>> {
        Err(Error::msg("not support buffer_from_host_ptr!"))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn info(&self) -> Result<DeviceInfo> {
        let mut device_info = DeviceInfo::default();
        device_info.name = self.info.name.clone();
        unsafe {
            cuda_bindings::cuMemGetInfo_v2(
                &mut device_info.memory.free as *mut usize,
                &mut device_info.memory.total as *mut usize,
            );
        }
        Ok(device_info)
    }
}

impl CudaBackendDevice {
    pub(super) fn init(&mut self, backend_ctx: Rc<RefCell<CudaBackendContext>>) -> Result<()> {
        if self.backend_ctx.is_some() {
            return Ok(());
        }

        self.backend_ctx = Some(backend_ctx);

        Ok(())
    }
}
