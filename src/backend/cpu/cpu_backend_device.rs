#[cfg(target_os = "linux")]
use libc::{_SC_PAGE_SIZE, _SC_PHYS_PAGES, sysconf};

use super::util::get_cpu_description;
use crate::backend::backend::{
    FemlBackendBuffer, FemlBackendBufferType, FemlBackendDevCaps, FemlBackendDevice,
    FemlBackendDeviceProps, FemlBackendDeviceType, FemlBackendEvent, FemlBackendReg,
};
use crate::backend::backend_trait::{FemlBackendDeviceInterface, FemlBackendRegInterface};
use crate::backend::cpu::cpu_context::FemlBackendCpuContext;
use crate::common::def::FemlGuid;
use crate::common::tensor::FemlTensor;
use crate::types::{FemlOpType, FemlStatus};
use crate::{backend::*, feml_abort, feml_error, utils};

fn feml_cpu_init() {
    //Todo imply feml cpu init
}

fn feml_backend_cpu_guid() -> FemlGuid {
    [0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89]
}

struct FemlCpuBackendRegImpl;

impl FemlBackendRegInterface for FemlCpuBackendRegImpl {
    fn get_name(&self, reg: &FemlBackendReg) -> *const str {
        "CPU Backend Registry"
    }

    fn get_device_count(&self, reg: &FemlBackendReg) -> usize {
        1
    }

    fn get_device(&self, reg: &FemlBackendReg, index: usize) -> Option<&FemlBackendDevice> {
        None
    }

    fn get_proc_address(&self, reg: &FemlBackendReg, name: &str) -> *const u8 {
        todo!()
    }
}

struct FemlCpuBackendDeviceImpl;

impl FemlBackendDeviceInterface for FemlCpuBackendDeviceImpl {
    fn get_name(&self, device: &FemlBackendDevice) -> &'static str {
        "CPU"
    }

    fn get_description(&self, device: &FemlBackendDevice) -> String {
        let cpu_desc = get_cpu_description().unwrap_or_else(|_| "Unknown".to_string());
        format!("Device: CPU: {}", cpu_desc)
    }

    fn get_memory(&self, device: &FemlBackendDevice) -> Result<(u64, u64), FemlStatus> {
        #[cfg(target_os = "linux")]
        {
            let pages = unsafe { sysconf(_SC_PHYS_PAGES) };
            let page_size = unsafe { sysconf(_SC_PAGE_SIZE) };

            if pages <= 0 || page_size <= 0 {
                return Err(FemlStatus::Failed);
            }

            let total = pages * page_size;
            let free = total; // 简单示例：将 total 作为 free

            Ok((free as u64, total as u64))
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(FemlStatus::Aborted)
        }
    }

    fn get_type(&self, device: &FemlBackendDevice) -> FemlBackendDeviceType {
        return FemlBackendDeviceType::CPU;
    }

    fn get_props(&self, device: &FemlBackendDevice, props: &mut FemlBackendDeviceProps) {
        props.name = self.get_name(device).to_owned();
        props.description = self.get_description(device);
        props.backend_type = self.get_type(device);
        self.get_memory(device)
            .map(|(free, total)| {
                props.free = free as u64;
                props.total = total as u64;
            })
            .unwrap_or_else(|_| {
                props.free = 0;
                props.total = 0;
            });
        props.caps = FemlBackendDevCaps {
            is_async: false,
            is_host_buffer: true,
            is_buffer_from_host_ptr: true,
            is_events: true,
        }
    }

    fn init_backend(&self, dev: &FemlBackendDevice, params: &Vec<u8>) {
        feml_cpu_init();
    }

    fn get_buffer_type(&self, device: &FemlBackendDevice) -> FemlBackendBufferType {
        todo!()
    }

    fn get_host_buffer_type(&self, device: &FemlBackendDevice) -> FemlBackendBufferType {
        todo!()
    }

    fn buffer_from_host_ptr(
        &self,
        device: &FemlBackendDevice,
        data: &Vec<u8>,
        max_tensor_size: usize,
    ) -> Option<FemlBackendBuffer> {
        todo!()
    }

    fn support_buft(&self, device: &FemlBackendDevice, buft: &FemlBackendBufferType) -> bool {
        todo!()
    }
    fn support_op(&self, device: &FemlBackendDevice, op: &mut FemlTensor) -> bool {
        todo!()
    }

    fn offload_op(&self, device: &FemlBackendDevice, op: &mut FemlTensor) -> bool {
        todo!()
    }

    fn event_new(&self, device: &FemlBackendDevice) -> FemlBackendEvent {
        todo!()
    }

    fn event_free(&self, device: &FemlBackendDevice, event: &FemlBackendEvent) {}

    fn event_synchronize(&self, device: &FemlBackendDevice, event: &FemlBackendEvent) {}
}
