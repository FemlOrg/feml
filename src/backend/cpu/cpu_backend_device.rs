#[cfg(target_os = "linux")]
use libc::{_SC_PAGE_SIZE, _SC_PHYS_PAGES, sysconf};

use super::util::get_cpu_description;
use crate::backend::backend::{
    FemlBackendBuffer, FemlBackendBufferType, FemlBackendDevCaps, FemlBackendDevice,
    FemlBackendDeviceProps, FemlBackendDeviceType, FemlBackendEvent,
};
use crate::backend::backend_trait::{FemlBackendDeviceInterface, FemlBackendRegInterface};
use crate::backend::cpu::api::feml_backend_cpu_init;
use crate::backend::cpu::cpu_backend::FemlBackendCpuImpl;
use crate::backend::cpu::cpu_context::FemlBackendCpuContext;
use crate::backend::cpu::cpu_register::{BackendFunction, BackendRegistry};
use crate::common::def::{FEML_DEFAULT_N_THREAD, FemlGuid};
use crate::common::tensor::FemlTensor;
use crate::types::{FemlOpType, FemlStatus};
use crate::{backend::*, feml_abort, feml_error, utils};

pub(crate) struct FemlCpuBackendDeviceImpl;

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
        feml_backend_cpu_init();
    }

    fn get_buffer_type(&self, device: &FemlBackendDevice) -> Option<FemlBackendBufferType> {
        None
    }

    fn get_host_buffer_type(&self, device: &FemlBackendDevice) -> Option<FemlBackendBufferType> {
        None
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
        false
    }

    fn event_new(&self, device: &FemlBackendDevice) -> Option<FemlBackendEvent> {
        None
    }

    fn event_free(&self, device: &FemlBackendDevice, event: &FemlBackendEvent) {}

    fn event_synchronize(&self, device: &FemlBackendDevice, event: &FemlBackendEvent) {}
}
