use super::super::backend::OpenclBackend;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub(crate) fn mul(
    backend: &OpenclBackend,
    src0: &Tensor,
    src1: &Tensor,
    dst: &Tensor,
) -> Result<()> {
    Err(Error::msg(format!("opencl: mul is not implemented yet")).context("in OpenclBackend::mul"))
}
