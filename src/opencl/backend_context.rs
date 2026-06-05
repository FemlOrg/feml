use super::backend_device::OpenclBackendDevice;
use crate::error::{Error, Result};
use ocl::{core::CommandQueueProperties, Event, Kernel, SpatialDims};
use std::collections::HashMap;

#[repr(usize)]
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ClKernelId {
    Mul = 0,
    MulRow = 1,
}

pub(super) enum OpenclGpuFamlily {
    Intel,
    Qualcomm,
    Unknown,
}

pub(super) struct OpenclBackendContext {
    pub(super) device: ocl::Device,
    pub(super) device_name: String,
    pub(super) context: ocl::Context,
    pub(super) queue: ocl::Queue,
    pub(super) wave_size: i32,
    pub(super) max_alloc_size: usize,
    pub(super) alignment: usize,
    pub(super) gpu_family: OpenclGpuFamlily,
    pub(super) kernels: Option<HashMap<ClKernelId, ocl::Kernel>>,
    pub(super) programs: Option<HashMap<&'static str, ocl::Program>>,
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclBackendDevice) -> Result<Self> {
        let mut props = CommandQueueProperties::ON_DEVICE_DEFAULT;
        #[cfg(feature = "opencl-profiling")]
        {
            props.profiling();
        }

        Ok(Self {
            device: device.device.clone(),
            device_name: device.device_name.clone(),
            context: device.context.clone(),
            queue: ocl::Queue::new(&device.context, device.device, Some(props))?,
            wave_size: 0,
            max_alloc_size: 0,
            alignment: 0,
            gpu_family: OpenclGpuFamlily::Unknown,
            kernels: None,
            programs: None,
        })
    }

    pub(super) fn load_cl_kernels(&mut self) -> Result<()> {
        if self.kernels.is_some() || self.programs.is_some() {
            return Ok(());
        }

        let mut kernels: HashMap<ClKernelId, ocl::Kernel> = HashMap::new();
        let mut programs: HashMap<&'static str, ocl::Program> = HashMap::new();

        // load mul kernel
        if !programs.contains_key("kernels/mul.cl") {
            let program = ocl::Program::builder()
                .src(include_str!("kernels/mul.cl"))
                .devices(self.device)
                .build(&self.context)?;
            programs.insert("mul", program.clone());

            let kernel_mul = ocl::Kernel::builder().program(&program).name("kernel_mul").build()?;
            kernels.insert(ClKernelId::Mul, kernel_mul);

            let kernel_mul_row =
                ocl::Kernel::builder().program(&program).name("kernel_mul_row").build()?;
            kernels.insert(ClKernelId::MulRow, kernel_mul_row);
        }

        self.kernels = Some(kernels);
        self.programs = Some(programs);

        Ok(())
    }

    pub(super) fn get_kernel(&self, kernel_id: ClKernelId) -> Result<&mut ocl::Kernel> {
        let ret = self.kernels.as_ref().unwrap().get_mut(&kernel_id).ok_or_else(|| {
            Error::msg(format!("Kernel {:?} not found in OpenCL backend context", kernel_id))
                .context("in OpenclBackendContext::get_kernel")
        });

        ret
    }

    pub(super) fn enqueue_ndrange_kernel(
        &self,
        kernel: &Kernel,
        global_dims: &SpatialDims,
        local_dims: &SpatialDims,
    ) -> Result<()> {
        kernel.set_default_global_work_size(*global_dims);
        kernel.set_default_local_work_size(*local_dims);
        #[cfg(feature = "opencl-profiling")]
        {
            let mut event = ocl::Event::empty();
            unsafe {
                kernel.cmd().enew(&mut event).enq()?;
            }
            event.wait_for()?;
        }
        #[cfg(not(feature = "opencl-profiling"))]
        {
            kernel.enq()?;
        }

        Ok(())
    }
}
