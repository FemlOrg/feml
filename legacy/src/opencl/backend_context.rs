use super::backend_device::OpenclBackendDevice;
use crate::error::{Error, Result};
use ocl::{core::CommandQueueProperties, Kernel};
use std::collections::HashMap;

#[repr(usize)]
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ClKernelId {
    Mul = 0,
    MulRow = 1,
}

#[allow(dead_code)]
pub(super) enum OpenclGpuFamlily {
    Intel,
    Qualcomm,
    Unknown,
}

pub(super) struct OpenclBackendContext {
    pub(super) device: ocl::Device,
    #[allow(dead_code)]
    pub(super) device_name: String,
    pub(super) context: ocl::Context,
    pub(super) queue: ocl::Queue,
    pub(super) wave_size: i32,
    #[allow(dead_code)]
    pub(super) max_alloc_size: usize,
    #[allow(dead_code)]
    pub(super) alignment: usize,
    pub(super) gpu_family: OpenclGpuFamlily,
    pub(super) kernels: Option<HashMap<ClKernelId, ocl::Kernel>>,
    pub(super) programs: Option<HashMap<&'static str, ocl::Program>>,
}

impl OpenclBackendContext {
    pub fn new(device: &OpenclBackendDevice) -> Result<Self> {
        #[allow(unused_mut)]
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

    pub(super) fn with_kernel<F>(&mut self, kernel_id: ClKernelId, f: F) -> Result<()>
    where
        F: FnOnce(&mut Kernel) -> Result<()>,
    {
        let kernel = self.kernels.as_mut().unwrap().get_mut(&kernel_id).ok_or_else(|| {
            Error::msg(format!("Kernel {:?} not found in OpenCL backend context", kernel_id))
                .context("in OpenclBackendContext::with_kernel")
        })?;
        f(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_id_values() {
        assert_eq!(ClKernelId::Mul as usize, 0);
        assert_eq!(ClKernelId::MulRow as usize, 1);
    }

    #[test]
    fn kernel_id_debug_display() {
        assert_eq!(format!("{:?}", ClKernelId::Mul), "Mul");
        assert_eq!(format!("{:?}", ClKernelId::MulRow), "MulRow");
    }

    #[test]
    fn kernel_id_equality() {
        assert_eq!(ClKernelId::Mul, ClKernelId::Mul);
        assert_ne!(ClKernelId::Mul, ClKernelId::MulRow);
    }

    #[test]
    fn kernel_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ClKernelId::Mul);
        set.insert(ClKernelId::MulRow);
        set.insert(ClKernelId::Mul);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn kernel_id_clone_copy() {
        let id = ClKernelId::Mul;
        let copy = id;
        assert_eq!(id, copy);
        let clone = id.clone();
        assert_eq!(id, clone);
    }
}
