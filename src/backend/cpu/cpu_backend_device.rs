use super::util::get_cpu_description;
use crate::backend::backend::{
    FemlBackendBuffer, FemlBackendBufferType, FemlBackendDevCaps, FemlBackendDevice,
    FemlBackendDeviceProps, FemlBackendDeviceType, FemlBackendEvent,
};
use crate::backend::backend_trait::FemlBackendDeviceInterface;
use crate::backend::cpu::api::feml_backend_cpu_init;
use crate::backend::cpu::cpu_buffer_type::FemlBackendCpuBufferTypeImpl;
use crate::common::tensor::FemlTensor;
use crate::types::FemlStatus;
#[cfg(target_os = "linux")]
use libc::{_SC_PAGE_SIZE, _SC_PHYS_PAGES, sysconf};

pub(crate) struct FemlCpuBackendDeviceImpl;

impl FemlBackendDeviceInterface for FemlCpuBackendDeviceImpl {
    fn get_name(&self, _device: &FemlBackendDevice) -> &'static str {
        "CPU"
    }

    fn get_description(&self, _device: &FemlBackendDevice) -> String {
        let cpu_desc = get_cpu_description().unwrap_or_else(|_| "Unknown".to_string());
        format!("Device: CPU: {}", cpu_desc)
    }

    fn get_memory(&self, _device: &FemlBackendDevice) -> Result<(u64, u64), FemlStatus> {
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

    fn get_type(&self, _device: &FemlBackendDevice) -> FemlBackendDeviceType {
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

    fn init_backend(&self, _dev: &FemlBackendDevice, _params: &Vec<u8>) {
        feml_backend_cpu_init();
    }

    fn get_buffer_type(&self, _device: &FemlBackendDevice) -> Option<FemlBackendBufferType> {
        Some(FemlBackendBufferType::new(Box::new(FemlBackendCpuBufferTypeImpl {}), None, None))
    }

    fn get_host_buffer_type(&self, _device: &FemlBackendDevice) -> Option<FemlBackendBufferType> {
        None
    }

    fn buffer_from_host_ptr(
        &self,
        _device: &FemlBackendDevice,
        _data: &Vec<u8>,
        _max_tensor_size: usize,
    ) -> Option<FemlBackendBuffer> {
        // Some(FemlBackendBuffer::new(
        //     Box::new(FemlBackendCpuBuffer {}),
        //     Arc::new(FemlBackendBufferType::new(Box::new(FemlBackendCpuBufferType {}), None, None)),
        //     Some(Box::new(context)),
        //     max_tensor_size,
        // ))
        None
    }

    fn support_buft(&self, _device: &FemlBackendDevice, _buft: &FemlBackendBufferType) -> bool {
        todo!()
    }
    fn support_op(&self, _device: &FemlBackendDevice, _op: &mut FemlTensor) -> bool {
        todo!()
    }

    fn offload_op(&self, _device: &FemlBackendDevice, _op: &mut FemlTensor) -> bool {
        false
    }

    fn event_new(&self, _device: &FemlBackendDevice) -> Option<FemlBackendEvent> {
        None
    }

    fn event_free(&self, _device: &FemlBackendDevice, _event: &FemlBackendEvent) {}

    fn event_synchronize(&self, _device: &FemlBackendDevice, _event: &FemlBackendEvent) {}
}
