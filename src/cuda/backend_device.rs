use super::backend_context::CudaBackendContext;
use crate::backend::{Backend, BackendDevice};
use crate::data_type::TensorOpType;
use crate::error::{Error, Result};
use cuda_core::{CudaContext, CudaStream};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Default)]
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

pub struct CudaBackendDevice {
    pub(super) info: CudaDeviceInfo,
    pub(super) context: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) backend_ctx: Option<Rc<RefCell<CudaBackendContext>>>,
}

impl BackendDevice for CudaBackendDevice {
    fn init_backend(&self) -> Result<Box<dyn Backend>> {
        todo!()
    }

    fn supports_op(&self, op_type: TensorOpType) -> Result<bool> {
        todo!()
    }

    fn offload_op(&self, tensor: crate::tensor::Tensor) -> crate::error::Result<bool> {
        todo!()
    }

    fn buffer_from_host_ptr(
        &self,
        ptr: &mut [u8],
        size: usize,
        max_tensor_size: usize,
    ) -> crate::error::Result<Box<dyn crate::backend::BackendBuffer>> {
        todo!()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        todo!()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        todo!()
    }

    fn info(&self) -> Result<crate::backend::DeviceInfo> {
        todo!()
    }
}

impl CudaBackendDevice {
    pub(super) fn init(&self, backend_ctx: Rc<RefCell<CudaBackendContext>>) -> Result<()> {
        if self.backend_ctx.is_some() {
            return Ok(());
        }

        self.backend_ctx = Some(backend_ctx);

        Ok(())
    }
}
