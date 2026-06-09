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

#[cfg(all(test, feature = "opencl"))]
mod tests {
    use super::*;
    use crate::backend::BackendBuffer;
    use std::any::Any;

    /// A minimal mock buffer for testing TensorStorage.
    #[derive(Clone)]
    struct MockBuffer(u8);

    impl BackendBuffer for MockBuffer {
        fn as_ptr(&self) -> crate::error::Result<*mut u8> {
            unimplemented!()
        }
        fn device(&self) -> crate::error::Result<Box<dyn crate::backend::BackendDevice>> {
            unimplemented!()
        }
        fn get_base(&self) -> crate::error::Result<*mut u8> {
            unimplemented!()
        }
        fn clear(&self, _value: u8) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn reset(&self) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn init_tensor(&self, _tensor: crate::tensor::Tensor, _offset: usize) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn memset_tensor(&self, _tensor: crate::tensor::Tensor, _value: u8, _offset: usize, _size: usize) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn set_tensor(&self, _tensor: crate::tensor::Tensor, _data: &mut [u8], _offset: usize, _size: usize) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn get_tensor(&self, _tensor: crate::tensor::Tensor, _data: &mut [u8], _offset: usize, _size: usize) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn copy_tensor(&self, _src: crate::tensor::Tensor, _dst: crate::tensor::Tensor) -> crate::error::Result<()> {
            unimplemented!()
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn get_usage(&self) -> crate::error::Result<crate::backend::BackendBufferUsage> {
            unimplemented!()
        }
        fn set_usage(&mut self, _usage: crate::backend::BackendBufferUsage) -> crate::error::Result<()> {
            unimplemented!()
        }
    }

    fn make_other_storage(offset: usize, size: usize) -> TensorStorage {
        TensorStorage::Other {
            buffer: Rc::new(MockBuffer(1)),
            offset,
            actual_size: size,
        }
    }

    #[test]
    fn as_opencl_returns_none_for_other() {
        let storage = make_other_storage(0, 0);
        assert!(storage.as_opencl().is_none());
    }

    #[test]
    fn offset_for_other_variant() {
        let storage = make_other_storage(42, 0);
        assert_eq!(storage.offset(), 42);
    }

    #[test]
    fn size_for_other_variant() {
        let storage = make_other_storage(0, 1024);
        assert_eq!(storage.size(), 1024);
    }

    #[test]
    fn buffer_returns_backend_buffer_ref() {
        let storage = make_other_storage(0, 0);
        let buf = storage.buffer();
        let _ = buf; // just verify it compiles and returns something
    }

    #[test]
    fn new_other_roundtrip() {
        let buf = Rc::new(MockBuffer(2));
        let storage = TensorStorage::new_other(buf, 100, 200);
        assert_eq!(storage.offset(), 100);
        assert_eq!(storage.size(), 200);
        assert!(storage.as_opencl().is_none());
    }

    #[test]
    fn new_other_buffer_identity() {
        let buf = Rc::new(MockBuffer(3));
        let storage = TensorStorage::new_other(buf.clone(), 0, 0);
        let returned = storage.buffer();
        let mock = returned.as_any().downcast_ref::<MockBuffer>();
        assert!(mock.is_some());
        assert_eq!(mock.unwrap().0, 3);
    }
}
