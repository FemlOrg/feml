use std::any::Any;
use std::sync::Arc;

use super::cpu_backend_reg_device::FemlCpuBackendRegDeviceImpl;
use crate::backend::api::feml_backend_reg_dev_get;
use crate::backend::backend::{FemlBackend, FemlBackendReg};
use crate::backend::cpu::cpu_backend::FemlBackendCpuImpl;
use crate::backend::cpu::cpu_context::FemlBackendCpuContext;
use crate::common::def::{FEML_BACKEND_API_VERION, FEML_DEFAULT_N_THREAD, FemlGuid};
use once_cell::sync::Lazy;

// TODO: init feml cpu init
pub fn feml_cpu_init() {}

pub(crate) fn feml_backend_cpu_guid() -> FemlGuid {
    [0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89]
}

pub fn feml_backend_cpu_init() -> Option<FemlBackend> {
    feml_cpu_init();
    let ctx: Option<Box<dyn Any>> =
        Some(Box::new(FemlBackendCpuContext::new(FEML_DEFAULT_N_THREAD)));
    Some(FemlBackend::new(
        feml_backend_cpu_guid(),
        Box::new(FemlBackendCpuImpl {}),
        Arc::new(feml_backend_reg_dev_get(feml_backend_cpu_reg(), 0).unwrap()),
        ctx,
    ))
}

pub fn feml_backend_cpu_reg() -> &'static Arc<FemlBackendReg> {
    feml_cpu_init();
    static CPU_REG: Lazy<Arc<FemlBackendReg>> = Lazy::new(|| {
        Arc::new(FemlBackendReg {
            interface: Box::new(FemlCpuBackendRegDeviceImpl),
            context: None,
            api_version: FEML_BACKEND_API_VERION,
        })
    });
    &CPU_REG
}
