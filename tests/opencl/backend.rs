#[cfg(feature = "opencl")]
mod opencl_backend {
    use feml::backend::BackendBufferUsage;
    use feml::context::{Context, ContextBuilder};
    use feml::data_type::DataType;
    use feml::registry::Registry;
    use feml::shape;
    use feml::tensor::Tensor;
    // ────────────────────────────────────────────────────────
    // Test 1: Backend register
    // ────────────────────────────────────────────────────────
    #[test]
    fn registry_find_ocl_reg() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let ocl_reg = reg.find("OpenCL");
        assert!(ocl_reg.is_some(), "find opencl register should succeed");
    }

    #[test]
    fn registry_ocl_device_count() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let ocl_reg = reg.find("OpenCL");
        assert!(ocl_reg.is_some(), "find opencl register should succeed");
        let device_count = ocl_reg.unwrap().device_count();
        assert_eq!(device_count, 1, "opencl backend device count{}", device_count);
    }

    #[test]
    fn registry_ocl_init_devices() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let ocl_reg = reg.find("OpenCL");
        assert!(ocl_reg.is_some(), "find opencl register should succeed");
        assert!(ocl_reg.unwrap().init_devices().is_ok(), "init device should succeed");
    }

    #[test]
    fn registry_ocl_device() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let ocl_reg = reg.find("OpenCL");
        assert!(ocl_reg.is_some(), "find opencl register should succeed");
        assert!(ocl_reg.unwrap().init_devices().is_ok(), "init device should succeed");
        let device = ocl_reg.unwrap().device(0);
        assert!(device.is_ok(), "get device should succeed");
    }

    #[test]
    fn registry_ocl_backend() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let ocl_reg = reg.find("OpenCL");
        assert!(ocl_reg.is_some(), "find opencl register should succeed");
        assert!(ocl_reg.unwrap().init_devices().is_ok(), "init device should succeed");
        let device = ocl_reg.unwrap().device(0);
        assert!(device.is_ok(), "get device should succeed");
        let backend = device.unwrap().init_backend();
        assert!(backend.is_ok(), "init backend should succeed");
    }

    #[test]
    fn registry_ocl_init_all() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        assert!(registry.unwrap().init_all().is_ok(), "registry init_all should succeed");
    }

    #[test]
    fn registry_ocl_open_device() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        assert!(
            registry.unwrap().open_device("OpenCL", 0).is_ok(),
            "open opencl backend should succeed"
        );
    }

    #[test]
    fn registry_ocl_open_backend() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        assert!(registry.as_ref().unwrap().init_all().is_ok(), "registry init_all should succeed");
        assert!(
            registry.unwrap().open_backend("OpenCL", 0).is_ok(),
            "open opencl backend should succeed"
        );
    }

    // ────────────────────────────────────────────────────────
    // Test 2: Buffer creation and basic operations
    // ────────────────────────────────────────────────────────
    #[test]
    fn create_backend_buffer() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        assert!(registry.as_ref().unwrap().init_all().is_ok(), "registry init_all should succeed");
        let backend = registry.as_ref().unwrap().open_backend("OpenCL", 0).unwrap();
        let backend_buffer = backend.create_buffer(1024, BackendBufferUsage::Any);
        assert!(backend_buffer.is_ok(), "create backend buffer should succeed");
        let usage = backend_buffer.unwrap().usage().unwrap();
        assert_eq!(usage, BackendBufferUsage::Any, "backend_buffer usage should be Any");
    }

    #[test]
    fn backend_buffer_init_tensor() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        assert!(registry.as_ref().unwrap().init_all().is_ok(), "registry init_all should succeed");
        let backend = registry.as_ref().unwrap().open_backend("OpenCL", 0).unwrap();
        let backend_buffer = backend.create_buffer(1024, BackendBufferUsage::Any).unwrap();

        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let tensor_shape = shape!(1, 2, 3);
        let tensor = ctx.new_tensor(DataType::U8, &tensor_shape).unwrap();
        assert!(
            backend_buffer.init_tensor(tensor, 0).is_ok(),
            "backend_buffer init_tensor should be succeed"
        );
    }

    #[test]
    fn backend_buffer_write_read() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        assert!(registry.as_ref().unwrap().init_all().is_ok(), "registry init_all should succeed");
        let backend = registry.as_ref().unwrap().open_backend("OpenCL", 0).unwrap();
        let backend_buffer = backend
            .create_buffer(256, BackendBufferUsage::Any)
            .expect("create backend buffer should succeed");
        let tensor_shape = shape!(1, 1, 1, 256);
        let mut ctx = Context::builder().tensor_pool_capacity(10).build();
        let tensor =
            ctx.new_tensor(DataType::U8, &tensor_shape).expect("create tensor should succeed");
        backend_buffer.init_tensor(tensor.clone(), 0);
        let mut input: Vec<u8> = (0u8..=255).collect();
        let mut test: Vec<u8> = Vec::new();
        test.resize(256, 0);
        backend_buffer
            .write(tensor.clone(), input.as_mut(), 0, 256)
            .expect("write tensor should succeed");
        backend_buffer
            .read(tensor.clone(), test.as_mut(), 0, 256)
            .expect("read tensor should succeed");
        for idx in 0..256 {
            assert_eq!(test[idx], input[idx]);
        }
    }
}
