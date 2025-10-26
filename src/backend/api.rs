use std::sync::Arc;

use crate::backend::backend::{
    FemlBackendBuffer, FemlBackendBufferType, FemlBackendDevice, FemlBackendReg,
};

pub fn feml_backend_reg_dev_get(
    reg: &Arc<FemlBackendReg>,
    index: usize,
) -> Option<FemlBackendDevice> {
    reg.interface.get_device(reg, index)
}

pub fn feml_backend_buffer_type_is_host(buffer_type: &FemlBackendBufferType) -> bool {
    buffer_type.interface.is_host(buffer_type)
}
pub fn feml_backend_buffer_is_host(buffer: &FemlBackendBuffer) -> bool {
    feml_backend_buffer_type_is_host(&*buffer.buffer_type.as_ref())
}
