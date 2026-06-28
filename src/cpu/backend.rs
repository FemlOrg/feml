use super::backend_buffers::CpuBackendBuffer;
use super::backend_context::CpuBackendContext;
use super::backend_device::CpuBackendDevice;
use super::backend_register::CpuBackendRegister;
use crate::backend::{Backend, BackendBuffer, BackendBufferUsage};
use crate::compute_graph::ComputeGraph;
use crate::context::Context;
use crate::data_type::{DataType, TensorOpType};
use crate::error::{Error, ErrorKind, Result};
use crate::tensor::Tensor;
use std::any::Any;

pub struct CpuBackend {
    #[allow(dead_code)]
    device: CpuBackendDevice,
    #[allow(dead_code)]
    context: CpuBackendContext,
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn graph_compute(&self, ctx: &Context, graph: &mut ComputeGraph) -> Result<()> {
        for node in graph.nodes().iter() {
            let tensor = ctx.get_tensor(*node)?;
            self.compute_forward(ctx, &tensor)?;
        }

        Ok(())
    }

    fn write_async(
        &self,
        tensor: Tensor,
        data: &mut [u8],
        offset: usize,
        size: usize,
    ) -> Result<()> {
        // let buffer = {
        //     let storage = tensor.storage()?;
        //     storage.as_cpu().ok_or_else(|| Error::msg("tensor storage is not CPU"))?.clone()
        // };
        // buffer.write(tensor, data, offset, size)
        todo!()
    }

    fn read_async(
        &self,
        tensor: Tensor,
        data: &mut [u8],
        offset: usize,
        size: usize,
    ) -> Result<()> {
        // let buffer = {
        //     let storage = tensor.storage()?;
        //     storage.as_cpu().ok_or_else(|| Error::msg("tensor storage is not CPU"))?.clone()
        // };
        // buffer.read(tensor, data, offset, size)
        todo!()
    }

    fn copy_async(&self, src: Tensor, dst: Tensor) -> Result<()> {
        // let buffer = {
        //     let storage = src.storage()?;
        //     storage.as_cpu().ok_or_else(|| Error::msg("source tensor storage is not CPU"))?.clone()
        // };
        // buffer.copy(src, dst)
        todo!()
    }

    fn create_buffer(
        &self,
        size: usize,
        usage: BackendBufferUsage,
    ) -> Result<Box<dyn BackendBuffer>> {
        Ok(Box::new(CpuBackendBuffer::new(size, usage)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl CpuBackend {
    pub fn new(device: CpuBackendDevice) -> Self {
        let context = CpuBackendContext::new();
        Self { device, context }
    }

    pub fn init() -> Result<Self> {
        let reg = CpuBackendRegister::init();
        let device = reg.cpu_device(0)?;
        Ok(Self::new(device))
    }

    fn compute_forward(&self, ctx: &Context, tensor: &Tensor) -> Result<()> {
        let src_tensor = tensor.src_tensor();
        match tensor.op_type() {
            TensorOpType::TensorOpMul => {
                if src_tensor.len() < 2 {
                    return Err(Error::msg("mul tensor requires two source tensors")
                        .context("in CpuBackend::compute_forward"));
                }

                let src0 = ctx.get_tensor(src_tensor[0])?;
                let src1 = ctx.get_tensor(src_tensor[1])?;
                self.mul(&src0, &src1, tensor)
            }
            _ => {
                let op_name = format!("{:?}", tensor.op_type());
                Err(Error::new(ErrorKind::UnsupportedBackendOp {
                    backend: "cpu",
                    op: "compute_forward",
                })
                .context(format!("unsupported op type: {}", op_name))
                .context("in CpuBackend::compute_forward"))
            }
        }
    }

    fn mul(&self, src0: &Tensor, src1: &Tensor, dst: &Tensor) -> Result<()> {
        if src0.dtype() != DataType::F32
            || src1.dtype() != DataType::F32
            || dst.dtype() != DataType::F32
        {
            return Err(Error::new(ErrorKind::UnsupportedDataTypeForOp {
                dtype: dst.dtype(),
                op: "cpu mul",
            }));
        }

        let src0_data = self.read_tensor_bytes(src0)?;
        let src1_data = self.read_tensor_bytes(src1)?;
        let mut dst_data = vec![0; dst.nbytes()];

        let src1_shape = *src1.shape();
        let dst_shape = *dst.shape();

        let src0_stride = {
            let stride = src0.stride();
            [stride[0], stride[1], stride[2], stride[3]]
        };
        let src1_stride = {
            let stride = src1.stride();
            [stride[0], stride[1], stride[2], stride[3]]
        };
        let dst_stride = {
            let stride = dst.stride();
            [stride[0], stride[1], stride[2], stride[3]]
        };

        let ne0 = dim(&dst_shape, 0);
        let ne1 = dim(&dst_shape, 1);
        let ne2 = dim(&dst_shape, 2);
        let ne3 = dim(&dst_shape, 3);

        let ne10 = dim(&src1_shape, 0);
        let ne11 = dim(&src1_shape, 1);
        let ne12 = dim(&src1_shape, 2);
        let ne13 = dim(&src1_shape, 3);

        for i3 in 0..ne3 {
            let i13 = i3 % ne13;
            for i2 in 0..ne2 {
                let i12 = i2 % ne12;
                for i1 in 0..ne1 {
                    let i11 = i1 % ne11;
                    for i0 in 0..ne0 {
                        let i10 = i0 % ne10;
                        let src0_offset = byte_offset(&src0_stride, i0, i1, i2, i3)?;
                        let src1_offset = byte_offset(&src1_stride, i10, i11, i12, i13)?;
                        let dst_offset = byte_offset(&dst_stride, i0, i1, i2, i3)?;

                        let value = read_f32(&src0_data, src0_offset, "src0")?
                            * read_f32(&src1_data, src1_offset, "src1")?;
                        write_f32(&mut dst_data, dst_offset, value, "dst")?;
                    }
                }
            }
        }

        self.write_tensor_bytes(dst, &mut dst_data)
    }

    fn read_tensor_bytes(&self, tensor: &Tensor) -> Result<Vec<u8>> {
        let mut data = vec![0; tensor.nbytes()];
        let buffer = {
            let storage = tensor.storage()?;
            storage
                .as_cpu()
                .ok_or_else(|| Error::msg("tensor storage is not CPU"))?
                .clone()
        };
        let size = data.len();
        buffer.read(tensor.clone(), &mut data, 0, size)?;
        Ok(data)
    }

    fn write_tensor_bytes(&self, tensor: &Tensor, data: &mut [u8]) -> Result<()> {
        let buffer = {
            let storage = tensor.storage()?;
            storage
                .as_cpu()
                .ok_or_else(|| Error::msg("tensor storage is not CPU"))?
                .clone()
        };
        buffer.write(tensor.clone(), data, 0, data.len())
    }
}

fn dim(shape: &crate::shape::Shape, index: usize) -> usize {
    if index < shape.rank {
        shape.dims[index]
    } else {
        1
    }
}

fn byte_offset(stride: &[usize; 4], i0: usize, i1: usize, i2: usize, i3: usize) -> Result<usize> {
    i0.checked_mul(stride[0])
        .and_then(|offset| i1.checked_mul(stride[1]).and_then(|delta| offset.checked_add(delta)))
        .and_then(|offset| i2.checked_mul(stride[2]).and_then(|delta| offset.checked_add(delta)))
        .and_then(|offset| i3.checked_mul(stride[3]).and_then(|delta| offset.checked_add(delta)))
        .ok_or_else(|| Error::msg("tensor byte offset overflow"))
}

fn read_f32(data: &[u8], offset: usize, name: &'static str) -> Result<f32> {
    let end = offset.checked_add(4).ok_or_else(|| Error::msg("f32 offset overflow"))?;
    let bytes = data
        .get(offset..end)
        .ok_or_else(|| {
            Error::msg(format!(
                "{name} f32 read is out of bounds: offset={offset}, len={}",
                data.len()
            ))
        })?;
    Ok(f32::from_ne_bytes(bytes.try_into().unwrap()))
}

fn write_f32(data: &mut [u8], offset: usize, value: f32, name: &'static str) -> Result<()> {
    let end = offset.checked_add(4).ok_or_else(|| Error::msg("f32 offset overflow"))?;
    let len = data.len();
    let dst = data
        .get_mut(offset..end)
        .ok_or_else(|| {
            Error::msg(format!(
                "{name} f32 write is out of bounds: offset={offset}, len={}",
                len
            ))
        })?;
    dst.copy_from_slice(&value.to_ne_bytes());
    Ok(())
}
