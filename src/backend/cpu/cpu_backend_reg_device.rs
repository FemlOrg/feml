use crate::backend::{
    backend::{FemlBackendDevice, FemlBackendReg},
    backend_trait::FemlBackendRegInterface,
};

struct FemlCpuBackendRegDeviceImpl;

impl FemlBackendRegInterface for FemlCpuBackendRegDeviceImpl {
    fn get_name(&self, reg: &FemlBackendReg) -> &'static str {
        "CPU"
    }
    fn get_device_count(&self, reg: &FemlBackendReg) -> usize {
        1
    }
    fn get_device(&self, reg: &FemlBackendReg, index: usize) -> Option<&FemlBackendDevice> {
        todo!()
    }
    fn get_proc_address(&self, reg: &FemlBackendReg, name: &str) -> *const u8 {
        todo!()
    }
}
