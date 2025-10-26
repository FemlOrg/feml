use crate::backend::cpu::cpu_backend_device::FemlCpuBackendDeviceImpl;
use crate::backend::cpu::cpu_context::FemlBackendCpuContext;
use crate::backend::cpu::cpu_register::{BackendFunction, BackendRegistry};
use crate::backend::{
    backend::{FemlBackendDevice, FemlBackendReg},
    backend_trait::FemlBackendRegInterface,
};
use crate::common::def::FEML_DEFAULT_N_THREAD;
use std::any::Any;
use std::sync::Arc;

pub(crate) struct FemlCpuBackendRegDeviceImpl;

impl FemlBackendRegInterface for FemlCpuBackendRegDeviceImpl {
    fn get_name(&self, _reg: &FemlBackendReg) -> &'static str {
        "CPU"
    }
    fn get_device_count(&self, _reg: &FemlBackendReg) -> usize {
        1
    }
    fn get_device(&self, reg: &Arc<FemlBackendReg>, _index: usize) -> Option<FemlBackendDevice> {
        let ctx: Option<Box<dyn Any>> =
            Some(Box::new(FemlBackendCpuContext::new(FEML_DEFAULT_N_THREAD)));
        Some(FemlBackendDevice::new(Box::new(FemlCpuBackendDeviceImpl {}), Arc::clone(reg), ctx))
    }
    fn get_proc_address(&self, _reg: &FemlBackendReg, name: &str) -> BackendFunction {
        let register = BackendRegistry::new();
        register.get_function(name).unwrap().clone()
    }
}
