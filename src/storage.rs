use std::rc::Rc;

use crate::backend::BackendBuffer;

#[derive(Clone)]
pub(crate) enum TensorStorage {
    None,
    Cpu {
        buffer: Rc<dyn BackendBuffer>,
        offset: usize,
    },
    #[cfg(feature = "cuda")]
    Cuda {
        buffer: Rc<dyn BackendBuffer>,
        offset: usize,
    },
    #[cfg(feature = "opencl")]
    OpenCL {
        buffer: Rc<dyn BackendBuffer>,
        ocl_buffer_idx: usize,
        offset: usize,
        acutal_size: usize,
    },
}
