use cuda_core::CudaContext;

use super::backend_context::CudaBackendContext;
use super::backend_device::{CudaBackendDevice, CudaDeviceInfo};
use super::common::CUDA_MAX_DEVICES;
use crate::backend::{BackendDevice, BackendRegister};
use crate::error::{Error, ErrorKind, Result};
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Once;

static INIT: Once = Once::new();
pub(super) static mut REG: Option<*const dyn BackendRegister> = None;

//In both CUDA and OpenCL, each device has its own CudaContext (or ocl::Context), whereas only a single global CudaBackendContext or OpenclBackendContext exists.
pub(super) struct CudaBackendRegister {
    pub(super) backend_ctx: Rc<RefCell<CudaBackendContext>>,
    pub(super) devices: RefCell<Vec<CudaBackendDevice>>,
}

impl BackendRegister for CudaBackendRegister {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_count(&self) -> usize {
        self.devices.borrow().len()
    }

    fn device(&self, index: usize) -> Result<Box<dyn BackendDevice>> {
        Ok(Box::new(self.cuda_device(index)?.clone()))
    }

    fn probe_devices(&self) -> Result<()> {
        let mut device_count: i32 = 0;
        unsafe {
            cuda_bindings::cuDeviceGetCount(&mut device_count as *mut i32);
        };
        if device_count > CUDA_MAX_DEVICES as i32 {
            return Err(Error::msg("exceed max devices!"));
        }

        let mut cuda_devices = self.devices.borrow_mut();
        for device_id in 0..device_count {
            let mut device_info = CudaDeviceInfo::default();
            let mut prop: cuda_bindings::CUdevprop_st;
            unsafe {
                cuda_bindings::cuDeviceGetProperties(&mut prop, device_id);
            }

            device_info.device_id = device_id;
            device_info.clock_rate = prop.clockRate;
            device_info.max_grid_size = prop.maxGridSize;
            device_info.max_threads_dim = prop.maxThreadsDim;
            device_info.max_threads_per_block = prop.maxThreadsPerBlock;
            device_info.shared_mem_per_block = prop.sharedMemPerBlock;
            device_info.total_constant_mem = prop.totalConstantMemory;
            device_info.regs_per_block = prop.regsPerBlock;
            device_info.warp_size = prop.SIMDWidth;

            let context = CudaContext::new(device_id as usize).unwrap();
            let stream = context.default_stream();
            device_info.name = context.device_name()? + &device_id.to_string();
            let cuda_device =
                CudaBackendDevice { info: device_info, context, stream, backend_ctx: None };

            cuda_devices.push(cuda_device);
        }

        Ok(())
    }

    fn init_devices(&self) -> Result<()> {
        let devices: Vec<CudaBackendDevice> = std::mem::take(&mut *self.devices.borrow_mut());
        let ctx = self.backend_ctx.clone();

        let valid_devices: Vec<CudaBackendDevice> =
            devices.into_iter().filter(|device| device.init(ctx.clone()).is_ok()).collect();

        *self.devices.borrow_mut() = valid_devices;

        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl CudaBackendRegister {
    pub fn init() -> &'static dyn BackendRegister {
        unsafe {
            INIT.call_once(|| {
                let reg = Self::new().into_inner();
                REG = Some(Box::into_raw(Box::new(reg)));
            });
            &*REG.unwrap()
        }
    }

    fn cuda_device(&self, index: usize) -> Result<&CudaBackendDevice> {
        self.devices.get(index).ok_or_else(|| {
            Error::new(ErrorKind::DeviceNotFound {
                backend: "cuda",
                index,
                count: self.devices.len(),
            })
        })
    }

    fn new() -> Rc<RefCell<Self>> {
        let reg = Rc::new(RefCell::new(Self {
            backend_ctx: Rc::new(RefCell::new(CudaBackendContext::new())),
            devices: RefCell::new(Vec::new()),
        }));
        reg.borrow_mut().backend_ctx.borrow_mut().register = Some(Rc::downgrade(&reg));
        reg
    }
}
