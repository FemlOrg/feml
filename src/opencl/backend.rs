use super::ops::mul::mul;
use crate::backend::{Backend, BackendBuffer, BackendBufferUsage};
use crate::compute_graph::ComputeGraph;
use crate::context::Context;
use crate::data_type::TensorOpType;
use crate::error::{Error, ErrorKind, Result};
use crate::tensor::Tensor;

use super::backend_buffer::OpenclBackendBuffer;
use super::backend_context::OpenclBackendContext;
use super::backend_register::OpenclBackendRegister;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
pub struct OpenclBackend {
    pub(super) backend_ctx: Rc<RefCell<OpenclBackendContext>>,
}

impl Backend for OpenclBackend {
    fn name(&self) -> &str {
        "opencl"
    }

    fn synchronize(&self) -> Result<()> {
        let ctx = self.backend_ctx.borrow();
        let event = ctx.queue.enqueue_marker(None::<()>)?;
        event.wait_for().map_err(ocl::Error::from)?;
        Ok(())
    }

    fn graph_compute(&self, ctx: &Context, graph: &mut ComputeGraph) -> Result<()> {
        for node in graph.nodes().iter() {
            let tensor = ctx.tensor(*node)?;
            self.compute_forward(ctx, &tensor)?;
        }

        Ok(())
    }

    fn write_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Err(Error::msg("opencl: set_tensor_async is not implemented yet")
            .context("in OpenclBackend::set_tensor_async"))
    }

    fn read_async(
        &self,
        _tensor: Tensor,
        _data: *mut u8,
        _offset: usize,
        _size: usize,
    ) -> Result<()> {
        Err(Error::msg("opencl: get_tensor_async is not implemented yet")
            .context("in OpenclBackend::get_tensor_async"))
    }

    fn copy_async(&self, _src: Tensor, _dst: Tensor) -> Result<()> {
        Err(Error::msg("opencl: copy_tensor_async is not implemented yet")
            .context("in OpenclBackend::copy_tensor_async"))
    }

    fn create_buffer(
        &self,
        size: usize,
        usage: BackendBufferUsage,
    ) -> Result<Box<dyn BackendBuffer>> {
        ocl::Buffer::<u8>::builder()
            .queue(self.backend_ctx.borrow().queue.clone())
            .len(size)
            .build()
            .map(|buffer| {
                Box::new(OpenclBackendBuffer::new(self.backend_ctx.clone(), buffer, usage, size))
                    as Box<dyn BackendBuffer>
            })
            .map_err(|e| Error::msg(format!("Failed to allocate OpenCL buffer: {}", e)))
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

        Ok(Box::new(Self { backend_ctx: device.backend_ctx.as_ref().unwrap().clone() }))
    }

    fn compute_forward(&self, ctx: &Context, tensor: &Tensor) -> Result<()> {
        let src_tensor = tensor.get_src_tensor();
        match tensor.get_op_type() {
            TensorOpType::TensorOpMul => {
                let src0 = ctx.get_tensor(src_tensor[0])?;
                let src1 = ctx.get_tensor(src_tensor[1])?;
                mul(self, &src0, &src1, tensor)
            }
            _ => {
                let op_name = format!("{:?}", tensor.get_op_type());
                Err(Error::new(ErrorKind::UnsupportedBackendOp {
                    backend: "opencl",
                    op: "compute_forward",
                })
                .context(format!("unsupported op type: {}", op_name))
                .context("in OpenclBackend::compute_forward"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_returns_opencl() {
        let reg = OpenclBackendRegister::init();
        assert!(reg.name() == "OpenCL");
    }

    #[test]
    fn backend_init_returns_result() {
        let result = OpenclBackend::init();
        if let Err(e) = &result {
            let msg = format!("{}", e);
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn unsupported_backend_op_error_contains_backend_name() {
        let err = Error::new(ErrorKind::UnsupportedBackendOp { backend: "opencl", op: "test_op" });
        let msg = format!("{}", err);
        assert!(msg.contains("opencl"));
        assert!(msg.contains("test_op"));
    }
}
