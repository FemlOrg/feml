#[cfg(feature = "cuda")]
mod cuda_backend {
    use std::rc::Rc;

    use feml::backend::{BackendBuffer, BackendBufferUsage};
    use feml::context::Context;
    use feml::cuda::backend_buffer::CudaBackendBuffer;
    use feml::data_type::DataType;
    use feml::registry::Registry;
    use feml::shape;
    use feml::storage::TensorStorage;
    // ────────────────────────────────────────────────────────
    // Test 1: Backend register, Backend, BackendDevice
    // ────────────────────────────────────────────────────────
    #[test]
    fn registry_find_cuda_reg() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let cuda_reg = reg.find("CUDA");
        assert!(cuda_reg.is_some(), "find CUDA register should succeed");
    }

    #[test]
    fn registry_cuda_device_count() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let cuda_reg = reg.find("CUDA");
        assert!(cuda_reg.is_some(), "find CUDA register should succeed");
        let device_count = cuda_reg.unwrap().device_count();
        assert_eq!(device_count, 1, "CUDA backend device count{}", device_count);
    }

    #[test]
    fn registry_cuda_init_devices() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let cuda_reg = reg.find("CUDA");
        assert!(cuda_reg.is_some(), "find CUDA register should succeed");
        assert!(cuda_reg.unwrap().init_devices().is_ok(), "init device should succeed");
        let device_count = cuda_reg.unwrap().device_count();
        assert_eq!(device_count, 1, "CUDA backend device count{}", device_count);
        let device = cuda_reg.unwrap().device(0);
        assert!(device.is_ok(), "registry device should succeed");
    }

    #[test]
    fn registry_cuda_device() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let cuda_reg = reg.find("CUDA");
        assert!(cuda_reg.is_some(), "find CUDA register should succeed");
        let device_count = cuda_reg.unwrap().device_count();
        assert_eq!(device_count, 1, "CUDA backend device count{}", device_count);
        let device = cuda_reg.unwrap().device(0);
        assert!(device.is_ok(), "registry device should succeed");
    }

    #[test]
    fn registry_cuda_init_all() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        assert!(reg.init_all().is_ok(), "registry init_all should succeed");
    }

    #[test]
    fn registry_cuda_open_backend() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        assert!(reg.init_all().is_ok(), "registry init_all should succeed");
        for idx in 0..reg.device_count() {
            let backend = reg.open_backend("CUDA", idx);
            assert!(backend.is_ok(), "registry openc cuda backend should succeed");
        }
    }

    #[test]
    fn registry_cuda_open_device() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        assert!(reg.init_all().is_ok(), "registry init_all should succeed");
        for idx in 0..reg.device_count() {
            let device = reg.open_device("CUDA", idx);
            assert!(device.is_ok(), "registry openc cuda backend should succeed");
        }
    }
    #[test]
    fn cuda_device_init_backend() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let cuda_reg = reg.find("CUDA");
        assert!(cuda_reg.is_some(), "find CUDA register should succeed");
        assert!(cuda_reg.unwrap().init_devices().is_ok(), "init device should succeed");
        let device_count = cuda_reg.unwrap().device_count();
        assert_eq!(device_count, 1, "CUDA backend device count{}", device_count);
        let device = cuda_reg.unwrap().device(0);
        assert!(device.is_ok(), "registry device should succeed");
        let backend = device.unwrap().init_backend();
        assert!(backend.is_ok(), "device.init_backend should succeed");
    }

    // ────────────────────────────────────────────────────────
    // Test 2: BackendBuffer
    // ────────────────────────────────────────────────────────
    #[test]
    fn cuda_backend_create_buffer() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        assert!(reg.init_all().is_ok(), "registry init_all should succeed");
        let backend = reg.open_backend("CUDA", 0);
        assert!(backend.is_ok(), "registry open cuda backend should succeed");
        let backend_buffer = backend.unwrap().create_buffer(1024, BackendBufferUsage::Any);
        assert!(backend_buffer.is_ok(), "backend.create_buffer should succeed");
        assert_eq!(
            backend_buffer.unwrap().usage().unwrap(),
            BackendBufferUsage::Any,
            "backend_buffer usage should be any"
        );
    }

    // TODO : test cuda_backend_buffer init_tensor, write & read tensor
}
