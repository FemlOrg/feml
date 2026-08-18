//! CUDA op kernels (strided mul/add broadcast, naive mul_mat, byte fill).

pub(crate) mod elementwise;
pub(crate) mod fill;
pub(crate) mod mul_mat;
