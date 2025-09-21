use std::mem;
use utils::log::*;

#[cfg(target_os = "linux")]
use libc::{sysconf, _SC_PAGE_SIZE, _SC_PHYS_PAGES};

use super::util::get_cpu_description;
use crate::backend::*;
use crate::common::tensor::FemlTensor;
use crate::types::FemlStatus;

struct FemlBackendCpuContext {
    pub n_threads: i32,
    pub data: Vec<u8>,
    pub abort_callback: Option<fn(Vec<u8>) -> bool>,
    pub abort_data: Vec<u8>,
    pub thread_pool: Option<ThreadPool>,
}

fn feml_cpu_init() {
    //Todo imply feml cpu init
}

fn feml_backend_cpu_guid() -> Vec<u8> {
    vec![
        0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc,
        0x89,
    ]
}

fn feml_backend_cpu_reg() -> FemlBackendReg {
    feml_cpu_init();

    FemlBackendReg {
        api_version: 1,
        interface: Box::new(FemlCpuBackendRegDevice {}),
        ctx: null_mut(),
    }
}

impl FemlBackendDeviceInterface for FemlCpuBackendDevice {
    fn get_name(&self, device: &FemlBackendDevice) -> String {
        "CPU".to_string()
    }
    fn get_description(&self, device: &FemlBackendDevice) -> String {
        get_cpu_description()
    }
    fn get_memory(&self, device: &FemlBackendDevice, free: &mut i64, total: &mut i64) {
        #[cfg(target_os = "linux")]
        {
            let pages = unsafe { sysconf(_SC_PHYS_PAGES) };
            let page_size = unsafe { sysconf(_SC_PAGE_SIZE) };

            if pages > 0 && page_size > 0 {
                total = (pages * page_size) as usize;
                free = total.copy();
            } else {
                total = 0;
                free = 0;
            }
        }
        feml_error!("get_memory() is not implemented on this platform");
        total = 0;
        free = 0;
    }
    fn get_type(&self, device: &FemlBackendDevice) -> FemlBackendDeviceType {
        return FemlBackendDeviceType::CPU;
    }
    fn get_props(&self, device: &FemlBackendDevice, props: &mut FemlBackendDeviceProps) {
        props.name = self.get_name(device);
        props.description = self.get_description(device);
        props.backend_type = self.get_type(device);
        self.get_memory(device, &mut props.free, &mut props.total);
        props.caps = FemlBackendDevCaps {
            is_async: false,
            is_host_buffer: false,
            buffer_fron_host_ptr: true,
            event: false,
        }
    }
    fn init_backend(&self, dev: &FemlBackendDevice) {
        feml_cpu_init();
        let ctx = FemlBackendCpuContext {
            n_threads: 1,
            data: Vec::new(),
            abort_callback: None,
            abort_data: Vec::new(),
            thread_pool: None,
        };
        unsafe {
            FemlBackend {
                guid: feml_backend_cpu_guid(),
                interface: Box::new(FemlCpuBackendInterface {}),
                ctx: ctx as *mut u8,
                device: feml_backend_reg_dev_get(feml_backend_cpu_reg(), 0),
            }
        }
    }
    fn get_buffer_type(&self, device: &FemlBackendDevice) -> FemlBackendBufferType {}
    fn get_host_buffer_type(&self, device: &FemlBackendDevice) -> FemlBackendBufferType {
        FemlBackendBufferType {}
    }
    fn buffer_from_host_ptr(
        &self,
        device: &FemlBackendDevice,
        &data: Vec<u8>,
        max_tensor_size: usize,
    ) -> FemlBackendBuffer {
        FemlBackendBuffer {}
    }
    fn support_buft(&self, device: &FemlBackendDevice, buft: &FemlBackendBufferType) -> bool {}
    fn supports_op(&self, device: &FemlBackendDevice, &mut op: FemlTensor) -> bool {
        let tensor1 = &mut op.src[0];
        let temsor2 = &mut op.src[1];
        if (op.op == FemlOpType::FemlOpTypeUnknown
            || op.op == FemlOpType::FemlOpReshape
            || op.op == FemlOpType::FemlOpView
            || op.op == FemlOpType::FemlOpPermute
            || op.op == FemlOpType::FemlOpTranspose)
        {
            return true;
        }

        return true;
    }
    fn offload_op(&self, device: &FemlBackendDevice, op: &FemlTensor) -> bool {
        return true;
    }
    fn event_new(&self, device: &FemlBackendDevice) -> FemlBackendEvent {
        FemlBackendEvent {}
    }
    fn event_free(&self, device: &FemlBackendDevice, event: &FemlBackendEvent) {}
    fn event_synchronize(&self, device: &FemlBackendDevice, event: &FemlBackendEvent) {}
}

impl FemlBackendRegInterface for FemlCpuBackendRegDevice {
    fn get_name(&self, reg: &FemlBackendReg) -> String {
        "CPU".to_string()
    }
    fn get_device_count(&self, reg: &FemlBackendReg) -> usize {
        1
    }
    fn get_device(&self, reg: &FemlBackendReg, index: usize) -> &mut FemlBackendDevice {
        if (0 == index) {
            feml_abort!("get_device() failed: invalid index");
        }
        static mut device: FemlBackendDevice =
            FemlBackendDevice { interface: Box::new(FemlCpuBackendDevice {}), reg: reg.clone() };
    }
    fn get_proc_address(&self, reg: &FemlBackendReg, name: &str) -> Option<&BackendFunction> {
        let register = BackendRegistry::new();
        register.get_function(name)
    }
}
