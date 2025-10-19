use crate::backend::cpu::cpu_backend_device::FemlCpuBackendDeviceImpl;
use crate::backend::cpu::cpu_context::FemlBackendCpuContext;
use crate::backend::cpu::cpu_register::{BackendFunction, BackendRegistry};
use crate::backend::{
    backend::{FemlBackendDevice, FemlBackendReg},
    backend_trait::FemlBackendRegInterface,
    cpu::cpu_backend::feml_cpu_init,
};
use crate::common::def::{FEML_BACKEND_API_VERSION, FEML_DEFAULT_N_THREAD};
use std::any::Any;
use std::rc::Rc;
struct FemlCpuBackendRegDeviceImpl;

impl FemlBackendRegInterface for FemlCpuBackendRegDeviceImpl {
    fn get_name(&self, reg: &FemlBackendReg) -> &'static str {
        "CPU"
    }
    fn get_device_count(&self, reg: &FemlBackendReg) -> usize {
        1
    }
    fn get_device(&self, reg: Rc<FemlBackendReg>, index: usize) -> Option<FemlBackendDevice> {
        let ctx: Option<Box<dyn Any>> =
            Some(Box::new(FemlBackendCpuContext::new(FEML_DEFAULT_N_THREAD)));
        Some(FemlBackendDevice::new(Box::new(FemlCpuBackendDeviceImpl {}), reg, ctx))
    }
    fn get_proc_address(&self, reg: &FemlBackendReg, name: &str) -> BackendFunction {
        let register = BackendRegistry::new();
        register.get_function(name).unwrap().clone()
    }
}

pub fn feml_backend_cpu_reg() -> Rc<FemlBackendReg> {
    feml_cpu_init();
    Rc::new(FemlBackendReg {
        interface: Box::new(FemlCpuBackendRegDeviceImpl),
        context: None,
        api_version: FEML_BACKEND_API_VERSION,
    })
}
