//! OpenCL mul kernel invocation (strided 4D broadcast).

use crate::backend::BackendInner;
use feml_core::error::{Error, Result};
use feml_core::shape::Shape;
use ocl::SpatialDims;

fn ocl_err(e: ocl::Error) -> Error {
    Error::backend("OpenCL", e.to_string())
}

pub(crate) struct TensorArg<'a> {
    pub buf: &'a ocl::Buffer<u8>,
    pub offset: usize,
    pub shape: &'a Shape,
    pub stride: &'a [usize; 4],
}

pub(crate) fn mul(
    inner: &mut BackendInner,
    a: &TensorArg<'_>,
    b: &TensorArg<'_>,
    c: &TensorArg<'_>,
) -> Result<()> {
    let kernel =
        inner.kernels.get_mut("kernel_mul").ok_or_else(|| Error::msg("mul: kernel not loaded"))?;

    let (ne00, ne01, ne02, ne03) =
        (a.shape[0] as i32, a.shape[1] as i32, a.shape[2] as i32, a.shape[3] as i32);
    let (nb00, nb01, nb02, nb03) =
        (a.stride[0] as u64, a.stride[1] as u64, a.stride[2] as u64, a.stride[3] as u64);
    let (ne10, ne11, ne12, ne13) =
        (b.shape[0] as i32, b.shape[1] as i32, b.shape[2] as i32, b.shape[3] as i32);
    let (nb10, nb11, nb12, nb13) =
        (b.stride[0] as u64, b.stride[1] as u64, b.stride[2] as u64, b.stride[3] as u64);
    let (ne0, ne1, ne2, ne3) =
        (c.shape[0] as i32, c.shape[1] as i32, c.shape[2] as i32, c.shape[3] as i32);
    let (nb0, nb1, nb2, nb3) =
        (c.stride[0] as u64, c.stride[1] as u64, c.stride[2] as u64, c.stride[3] as u64);

    kernel.set_arg(0, a.buf).map_err(ocl_err)?;
    kernel.set_arg(1, a.offset as u64).map_err(ocl_err)?;
    kernel.set_arg(2, b.buf).map_err(ocl_err)?;
    kernel.set_arg(3, b.offset as u64).map_err(ocl_err)?;
    kernel.set_arg(4, c.buf).map_err(ocl_err)?;
    kernel.set_arg(5, c.offset as u64).map_err(ocl_err)?;
    kernel.set_arg(6, ne00).map_err(ocl_err)?;
    kernel.set_arg(7, ne01).map_err(ocl_err)?;
    kernel.set_arg(8, ne02).map_err(ocl_err)?;
    kernel.set_arg(9, ne03).map_err(ocl_err)?;
    kernel.set_arg(10, nb00).map_err(ocl_err)?;
    kernel.set_arg(11, nb01).map_err(ocl_err)?;
    kernel.set_arg(12, nb02).map_err(ocl_err)?;
    kernel.set_arg(13, nb03).map_err(ocl_err)?;
    kernel.set_arg(14, ne10).map_err(ocl_err)?;
    kernel.set_arg(15, ne11).map_err(ocl_err)?;
    kernel.set_arg(16, ne12).map_err(ocl_err)?;
    kernel.set_arg(17, ne13).map_err(ocl_err)?;
    kernel.set_arg(18, nb10).map_err(ocl_err)?;
    kernel.set_arg(19, nb11).map_err(ocl_err)?;
    kernel.set_arg(20, nb12).map_err(ocl_err)?;
    kernel.set_arg(21, nb13).map_err(ocl_err)?;
    kernel.set_arg(22, ne0).map_err(ocl_err)?;
    kernel.set_arg(23, ne1).map_err(ocl_err)?;
    kernel.set_arg(24, ne2).map_err(ocl_err)?;
    kernel.set_arg(25, ne3).map_err(ocl_err)?;
    kernel.set_arg(26, nb0).map_err(ocl_err)?;
    kernel.set_arg(27, nb1).map_err(ocl_err)?;
    kernel.set_arg(28, nb2).map_err(ocl_err)?;
    kernel.set_arg(29, nb3).map_err(ocl_err)?;

    let nth = 64.min(ne0.max(1) as usize);
    let global: [usize; 3] =
        [(ne01.max(1) as usize) * nth, ne02.max(1) as usize, ne03.max(1) as usize];
    let local: [usize; 3] = [nth, 1, 1];
    kernel.set_default_global_work_size(SpatialDims::from(global));
    kernel.set_default_local_work_size(SpatialDims::from(local));

    unsafe { kernel.enq() }.map_err(ocl_err)
}
