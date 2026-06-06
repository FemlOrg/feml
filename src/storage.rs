use crate::backend::BackendBuffer;
use crate::cpu::backend_buffers::CpuBackendBuffer;
use crate::opencl::backend_buffer::OpenclBackendBuffer;
use std::rc::Rc;

#[derive(Clone)]
pub(crate) enum TensorStorage {
    Cpu { buffer: Rc<CpuBackendBuffer>, offset: usize, actual_size: usize },
    Opencl { buffer: Rc<OpenclBackendBuffer>, offset: usize, actual_size: usize },
    Other { buffer: Rc<dyn BackendBuffer>, offset: usize, actual_size: usize },
}

impl TensorStorage {
    pub(crate) fn offset(&self) -> usize {
        match self {
            Self::Cpu { offset, .. } => *offset,
            Self::Opencl { offset, .. } => *offset,
            Self::Other { offset, .. } => *offset,
        }
    }

    pub(crate) fn size(&self) -> usize {
        match self {
            Self::Cpu { actual_size, .. } => *actual_size,
            Self::Opencl { actual_size, .. } => *actual_size,
            Self::Other { actual_size, .. } => *actual_size,
        }
    }

    pub(crate) fn buffer(&self) -> &dyn BackendBuffer {
        match self {
            Self::Cpu { buffer, .. } => buffer.as_ref(),
            Self::Opencl { buffer, .. } => buffer.as_ref(),
            Self::Other { buffer, .. } => buffer.as_ref(),
        }
    }

    pub(crate) fn as_cpu(&self) -> Option<&CpuBackendBuffer> {
        match self {
            Self::Cpu { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }

    pub(crate) fn as_opencl(&self) -> Option<&OpenclBackendBuffer> {
        match self {
            Self::Opencl { buffer, .. } => Some(buffer.as_ref()),
            _ => None,
        }
    }
}
