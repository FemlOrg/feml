use std::sync::Arc;

use crate::backend::backend::{FemlBackendDevice, FemlBackendReg};

pub fn feml_backend_reg_dev_get(
    reg: &Arc<FemlBackendReg>,
    index: usize,
) -> Option<FemlBackendDevice> {
    reg.interface.get_device(reg, index)
}
