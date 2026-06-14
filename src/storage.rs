use crate::backend::BackendBuffer;
#[cfg(feature = "cpu")]
use crate::cpu::backend_buffers::CpuBackendBuffer;
#[cfg(feature = "cuda")]
use crate::cuda::backend_buffer::CudaBackendBuffer;
#[cfg(feature = "opencl")]
use crate::opencl::backend_buffer::OpenclBackendBuffer;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) enum TensorStorage {
    #[cfg(feature = "cpu")]
    Cpu { buffer: Rc<CpuBackendBuffer>, offset: usize, actual_size: usize },
    #[cfg(feature = "opencl")]
    Opencl { buffer: Rc<OpenclBackendBuffer>, offset: usize, actual_size: usize },
    #[cfg(feature = "cuda")]
    Cuda { buffer: Rc<CudaBackendBuffer>, offset: usize, actual_size: usize },
    #[allow(dead_code)]
    Other { buffer: Rc<dyn BackendBuffer>, offset: usize, actual_size: usize },
}

impl TensorStorage {
    #[cfg(feature = "cpu")]
    pub(crate) fn new_cpu(buffer: Rc<CpuBackendBuffer>, offset: usize, actual_size: usize) -> Self {
        Self::Cpu { buffer, offset, actual_size }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn new_cuda(
        buffer: Rc<CudaBackendBuffer>,
        offset: usize,
        actual_size: usize,
    ) -> Self {
        Self::Cuda { buffer, offset, actual_size }
    }

    #[cfg(feature = "opencl")]
    pub(crate) fn new_opencl(
        buffer: Rc<OpenclBackendBuffer>,
        offset: usize,
        actual_size: usize,
    ) -> Self {
        Self::Opencl { buffer, offset, actual_size }
    }

    #[allow(dead_code)]
    pub(crate) fn new_other(
        buffer: Rc<dyn BackendBuffer>,
        offset: usize,
        actual_size: usize,
    ) -> Self {
        Self::Other { buffer, offset, actual_size }
    }

    pub(crate) fn offset(&self) -> usize {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu { offset, .. } => *offset,
            #[cfg(feature = "opencl")]
            Self::Opencl { offset, .. } => *offset,
            #[cfg(feature = "cuda")]
            Self::Cuda { offset, .. } => *offset,
            Self::Other { offset, .. } => *offset,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn size(&self) -> usize {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu { actual_size, .. } => *actual_size,
            #[cfg(feature = "opencl")]
            Self::Opencl { actual_size, .. } => *actual_size,
            #[cfg(feature = "cuda")]
            Self::Cuda { actual_size, .. } => *actual_size,
            Self::Other { actual_size, .. } => *actual_size,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn buffer(&self) -> &dyn BackendBuffer {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu { buffer, .. } => buffer.as_ref(),
            #[cfg(feature = "opencl")]
            Self::Opencl { buffer, .. } => buffer.as_ref(),
            #[cfg(feature = "cuda")]
            Self::Cuda { buffer, .. } => buffer.as_ref(),
            Self::Other { buffer, .. } => buffer.as_ref(),
        }
    }

    #[cfg(feature = "cpu")]
    pub(crate) fn as_cpu(&self) -> Option<&CpuBackendBuffer> {
        match self {
            Self::Cpu { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }

    #[cfg(feature = "opencl")]
    pub(crate) fn as_opencl(&self) -> Option<&OpenclBackendBuffer> {
        match self {
            Self::Opencl { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda(&self) -> Option<&CudaBackendBuffer> {
        match self {
            Self::Cuda { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }
}
