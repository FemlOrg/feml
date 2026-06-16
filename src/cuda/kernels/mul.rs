use crate::cuda::backend::CudaBackend;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
pub(crate) fn mul(backend: &CudaBackend, src0: &Tensor, src1: &Tensor, dst: &Tensor) -> Result<()> {
    todo!()
}
