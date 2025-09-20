use crate::common::tensor::FemlTensor;

// use trait to implment backend dynamic polymorphism
pub trait FemlBackendBufferTypeInterface {
    fn get_name(&self, buffer_type: &FemlBackendBufferType) -> *const str;

    // allocate a buffer of this type
    fn alloc_buffer(&self, buffer_type: &FemlBackendBufferType, size: usize) -> FemlBackendBuffer;

    // tensor alignment
    fn get_alignment(&self, buffer_type: &FemlBackendBufferType) -> usize;

    // max buffer size that can be allocated (defaults to SIZE_MAX)
    fn get_max_size(&self, buffer_type: &FemlBackendBufferType) -> usize;

    // data size needed to allocate the tensor, including padding (defaults to feml_nbytes)
    fn get_alloc_size(&self, buffer_type: &FemlBackendBufferType, tensor: &mut FemlTensor);

    // check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
    fn is_host(&self, buffer_type: &FemlBackendBufferType) -> bool;
}

pub struct FemlBackendBufferType {
    pub interface: Box<dyn FemlBackendBufferTypeInterface>,
    pub device: FemlBackendDevice,
    pub context: *mut u8,
}

pub trait FemlBackendBufferInterface {
    // free the buffer
    fn free_buffer(&self, buffer: &FemlBackendBuffer);

    // base address of the buffer
    fn get_base(&self, buffer: &FemlBackendBuffer);

    //
}

pub struct FemlBackendBuffer;

pub struct FemlBackendDevice;
