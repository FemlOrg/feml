use std::cmp::min;

use super::super::backend::OpenclBackend;
use super::super::backend_context::ClKernelId;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use ocl::SpatialDims;

pub(crate) fn mul(
    backend: &OpenclBackend,
    src0: &Tensor,
    src1: &Tensor,
    dst: &Tensor,
) -> Result<()> {
    let mut backend_ctx = backend.backend_ctx.borrow_mut();

    let ne00 = src0.get_shape()[0];
    let ne01 = src0.get_shape()[1];
    let ne02 = src0.get_shape()[2];
    let ne03 = src0.get_shape()[3];

    let nb00 = src0.get_stride()[0];
    let nb01 = src0.get_stride()[1];
    let nb02 = src0.get_stride()[2];
    let nb03 = src0.get_stride()[3];

    let ne10 = src1.get_shape()[0];
    let ne11 = src1.get_shape()[1];
    let ne12 = src1.get_shape()[2];
    let ne13 = src1.get_shape()[3];

    let nb10 = src1.get_stride()[0];
    let nb11 = src1.get_stride()[1];
    let nb12 = src1.get_stride()[2];
    let nb13 = src1.get_stride()[3];

    let ne0 = dst.get_shape()[0];
    let ne1 = dst.get_shape()[1];
    let ne2 = dst.get_shape()[2];
    let ne3 = dst.get_shape()[3];

    let nb0 = dst.get_stride()[0];
    let nb1 = dst.get_stride()[1];
    let nb2 = dst.get_stride()[2];
    let nb3 = dst.get_stride()[3];

    let src0_storage = src0.get_extra_storage()?;
    let src1_storage = src1.get_extra_storage()?;
    let dst_storage = dst.get_extra_storage()?;

    let offset0 = src0_storage.offset() + src0.get_view_offset();
    let offset1 = src1_storage.offset() + src1.get_view_offset();
    let offsetd = dst_storage.offset() + dst.get_view_offset();

    let src0_buffer = src0_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("source 0 storage is not OpenCL type"))?;
    let src1_buffer = src1_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("source 1 storage is not OpenCL type"))?;
    let dst_buffer = dst_storage
        .as_opencl()
        .ok_or_else(|| Error::msg("destination storage is not OpenCL type"))?;

    let nth = min(64, ne0);
    let global_work_size: [usize; 3] = [ne01 * nth, ne02, ne03];
    let local_work_size: [usize; 3] = [nth, 1, 1];
    let global_work_dims = SpatialDims::from(global_work_size);
    let local_work_dims = SpatialDims::from(local_work_size);

    backend_ctx.with_kernel(ClKernelId::Mul, |kernel| {
        kernel.set_arg(0, &src0_buffer.buffer)?;
        kernel.set_arg(1, offset0)?;
        kernel.set_arg(2, &src1_buffer.buffer)?;
        kernel.set_arg(3, offset1)?;
        kernel.set_arg(4, &dst_buffer.buffer)?;
        kernel.set_arg(5, offsetd)?;
        kernel.set_arg(6, ne00)?;
        kernel.set_arg(7, ne01)?;
        kernel.set_arg(8, ne02)?;
        kernel.set_arg(9, ne03)?;
        kernel.set_arg(10, nb00)?;
        kernel.set_arg(11, nb01)?;
        kernel.set_arg(12, nb02)?;
        kernel.set_arg(13, nb03)?;
        kernel.set_arg(14, ne10)?;
        kernel.set_arg(15, ne11)?;
        kernel.set_arg(16, ne12)?;
        kernel.set_arg(17, ne13)?;
        kernel.set_arg(18, nb10)?;
        kernel.set_arg(19, nb11)?;
        kernel.set_arg(20, nb12)?;
        kernel.set_arg(21, nb13)?;
        kernel.set_arg(22, ne0)?;
        kernel.set_arg(23, ne1)?;
        kernel.set_arg(24, ne2)?;
        kernel.set_arg(25, ne3)?;
        kernel.set_arg(26, nb0)?;
        kernel.set_arg(27, nb1)?;
        kernel.set_arg(28, nb2)?;
        kernel.set_arg(29, nb3)?;

        kernel.set_default_global_work_size(global_work_dims);
        kernel.set_default_local_work_size(local_work_dims);

        unsafe {
            kernel.enq()?;
        }
        Ok(())
    })?;

    Ok(())
}
