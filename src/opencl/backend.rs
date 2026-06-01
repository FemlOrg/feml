use crate::backend::Backend;

use super::backend_context::OpenclBackendContext;
use super::backend_device::OpenclBackendDevice;
use super::backend_register::OpenclBackendRegister;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
pub struct OpenclBackend {
    context: Rc<RefCell<OpenclBackendContext>>,
}

impl Backend for OpenclBackend {
    type Device = OpenclDevice;

    fn name(&self) -> &str {
        "opencl"
    }

    fn synchronize(&self) -> Result<()> {
        let event = self.context.queue.enqueue_marker(None::<()>)?;
        event.wait_for().map_err(ocl::Error::from)?;
        Ok(())
    }

    fn graph_compute(&self, _graph: &mut ComputeGraph) -> Result<()> {
        Err(Error::new(ErrorKind::UnsupportedBackendOp { backend: "opencl", op: "graph_compute" }))
    }

    fn memcpy_async(&self, _dst: *mut u8, _src: *const u8, _size: usize) -> Result<()> {
        Ok(())
    }

    fn set_tensor_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Ok(())
    }

    fn get_tensor_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Ok(())
    }

    fn copy_tensor_async(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl OpenclBackend {
    pub fn init() -> Result<Box<dyn Backend>> {
        let reg =
            OpenclBackendRegister::init().as_any().downcast_ref::<OpenclBackendRegister>().unwrap();
        let device = reg.opencl_device(0)?;

        Ok(Box::new(Self { context: device.backend_ctx.as_ref().unwrap().clone() }))
    }
}
