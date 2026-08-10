#[cfg(feature = "opencl")]
mod opencl_backend {
    // use feml::backend::{Backend, BackendBuffer, BackendBufferUsage};
    // use feml::context::{Context, ContextBuilder};
    // use feml::data_type::DataType;
    // use feml::opencl::backend::OpenclBackend;
    // use feml::opencl::backend_register::OpenclBackendRegister;
    use feml::registry::Registry;
    // use feml::shape;
    // use feml::shape::Shape;
    // use feml::tensor::Tensor;

    /// Helper: create a context with OpenCL backend auto-initialized
    // fn context_with_opencl() -> Option<Context> {
    //     let reg = OpenclBackendRegister::init();
    //     if reg.device_count() == 0 {
    //         eprintln!("[SKIP] No OpenCL devices found");
    //         return None;
    //     }
    //     let mut ctx = Context::builder().tensor_pool_capacity(16).build();
    //     Some(ctx)
    // }

    // /// Helper: get backend from context (or create one)
    // fn opencl_backend() -> Option<Box<dyn Backend>> {
    //     let result = OpenclBackend::init();
    //     match result {
    //         Ok(backend) => {
    //             println!("OpenCL backend: {}", backend.name());
    //             Some(backend)
    //         }
    //         Err(e) => {
    //             eprintln!("[SKIP] Cannot init OpenCL backend: {}", e);
    //             None
    //         }
    //     }
    // }

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
        // let device = ocl_reg.unwrap().device(0);
        // assert!(device.is_ok(), "registry device should succeed");
    }

    #[test]
    fn registry_ocl_device() {
        let registry = Registry::discover();
        assert!(registry.is_ok(), "registry discover should succeed");
        let reg = registry.ok().unwrap();
        let ocl_reg = reg.find("OpenCL");
        assert!(ocl_reg.is_some(), "find opencl register should succeed");
        let device = ocl_reg.unwrap().device(0);
        assert!(device.is_ok(), "registry device should succeed");
    }

    // #[test]
    // fn backend_init_and_name() {
    //     let Some(backend) = opencl_backend() else { return };
    //     assert_eq!(backend.name(), "opencl");
    // }

    // #[test]
    // fn backend_synchronize() {
    //     let Some(backend) = opencl_backend() else { return };
    //     assert!(backend.synchronize().is_ok(), "synchronize should succeed");
    // }

    // #[test]
    // fn backend_as_any_downcast() {
    //     let Some(backend) = opencl_backend() else { return };
    //     assert!(backend.as_any().is::<OpenclBackend>());
    // }

    // // ────────────────────────────────────────────────────────
    // // Test 2: Buffer creation and basic operations
    // // ────────────────────────────────────────────────────────

    // #[test]
    // fn buffer_create_and_capacity() {
    //     let Some(backend) = opencl_backend() else { return };
    //     let buf =
    //         backend.create_buffer(4096, BackendBufferUsage::Any).expect("should create buffer");

    //     assert!(buf.size() > 0, "buffer should have positive size");
    //     assert_eq!(buf.get_usage().unwrap(), BackendBufferUsage::Any);
    // }

    // #[test]
    // fn buffer_write_read_roundtrip() {
    //     let Some(backend) = opencl_backend() else { return };
    //     let buf =
    //         backend.create_buffer(256, BackendBufferUsage::Compute).expect("should create buffer");

    //     // Write data
    //     let input: Vec<u8> = (0u8..=255).collect();
    //     assert!(
    //         buf.write(Tensor::new(), &mut input.clone(), 0, 256).is_ok() || true,
    //         "write may succeed"
    //     );
    // }

    // #[test]
    // fn buffer_usage_roundtrip() {
    //     let Some(backend) = opencl_backend() else { return };
    //     let mut buf =
    //         backend.create_buffer(64, BackendBufferUsage::Any).expect("should create buffer");

    //     assert_eq!(buf.get_usage().unwrap(), BackendBufferUsage::Any);
    //     buf.set_usage(BackendBufferUsage::Weights).unwrap();
    //     assert_eq!(buf.get_usage().unwrap(), BackendBufferUsage::Weights);
    //     buf.set_usage(BackendBufferUsage::Compute).unwrap();
    //     assert_eq!(buf.get_usage().unwrap(), BackendBufferUsage::Compute);
    // }

    // #[test]
    // fn buffer_fill() {
    //     let Some(backend) = opencl_backend() else { return };
    //     let buf =
    //         backend.create_buffer(128, BackendBufferUsage::Compute).expect("should create buffer");

    //     assert!(buf.fill(Tensor::new(), 0xAB, 0, 128).is_ok());
    // }

    // // ────────────────────────────────────────────────────────
    // // Test 3: Tensor creation in context
    // // ────────────────────────────────────────────────────────

    // #[test]
    // fn tensor_create_in_context() {
    //     let Some(mut ctx) = context_with_opencl() else { return };
    //     let shape = shape![2, 3, 1, 1];
    //     let t = ctx.new_tensor(DataType::F32, &shape).expect("should create tensor");

    //     assert_eq!(t.dtype(), DataType::F32);
    //     assert_eq!(&*t.shape(), &shape);
    // }

    // #[test]
    // fn tensor_set_op_and_sources() {
    //     let Some(mut ctx) = context_with_opencl() else { return };
    //     let shape = shape![2, 2, 1, 1];

    //     let a = ctx.new_tensor(DataType::F32, &shape).unwrap();
    //     let b = ctx.new_tensor(DataType::F32, &shape).unwrap();

    //     // Set up mul: c = a * b
    //     let c = ctx.new_tensor(DataType::F32, &shape).unwrap();
    //     c.set_op_type(feml::data_type::TensorOpType::TensorOpMul);
    //     c.set_src_tensor(a.tensor_id());
    //     c.set_src_tensor(b.tensor_id());

    //     assert_eq!(c.op_type(), feml::data_type::TensorOpType::TensorOpMul);
    //     assert_eq!(c.src_tensor().len(), 2);
    // }

    // // ────────────────────────────────────────────────────────
    // // Test 4: End-to-end Mul operator
    // // ────────────────────────────────────────────────────────

    // #[test]
    // fn end_to_end_mul_operator() {
    //     let Some(backend) = opencl_backend() else { return };

    //     // 1. Create buffer and write input data
    //     let n_elements: usize = 4;
    //     let n_bytes = n_elements * std::mem::size_of::<f32>();
    //     let buf =
    //         backend.create_buffer(n_bytes * 3, BackendBufferUsage::Compute).expect("create buffer");

    //     let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    //     let b_data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];

    //     // 2. Create tensors in context
    //     let Some(mut ctx) = context_with_opencl() else { return };
    //     let shape = shape![n_elements, 1, 1, 1];

    //     let a = ctx.new_tensor(DataType::F32, &shape).unwrap();
    //     let b = ctx.new_tensor(DataType::F32, &shape).unwrap();
    //     let c = a.mul(b).unwrap();

    //     // 3. Bind tensors to buffer (init_tensor sets up storage)
    //     buf.init_tensor(a.clone(), 0).unwrap();
    //     buf.init_tensor(b.clone(), n_bytes).unwrap();
    //     buf.init_tensor(c.clone(), n_bytes * 2).unwrap();

    //     // 4. Upload data to device
    //     let a_bytes = bytemuck::cast_slice(&a_data).to_vec();
    //     buf.write(a.clone(), &mut a_bytes.clone(), 0, n_bytes).unwrap();

    //     let b_bytes = bytemuck::cast_slice(&b_data).to_vec();
    //     buf.write(b.clone(), &mut b_bytes.clone(), 0, n_bytes)?;

    //     // 6. Execute via compute_forward (direct call, bypassing graph)
    //     let opencl = backend
    //         .as_any()
    //         .downcast_ref::<OpenclBackend>()
    //         .expect("should downcast to OpenclBackend");

    //     // Build compute graph and execute
    //     let mut graph = ctx.new_graph(1)?;
    //     graph.build_forward(&ctx, c.tensor_id(), false).unwrap();
    //     backend.graph_compute(&ctx, &mut graph).unwrap();
    //     backend.synchronize().unwrap();

    //     // 7. Read back result
    //     let mut out_bytes = vec![0u8; n_bytes];
    //     buf.read(c.clone(), &mut out_bytes, 0, n_bytes).unwrap();
    //     let out_f32: &[f32] = bytemuck::cast_slice(&out_bytes);

    //     // 8. Verify: expected = a * b element-wise
    //     for i in 0..n_elements {
    //         let expected = a_data[i] * b_data[i];
    //         assert!(
    //             (out_f32[i] - expected).abs() < 0.001,
    //             "mul({}, {}) = {} != expected {}",
    //             a_data[i],
    //             b_data[i],
    //             out_f32[i],
    //             expected
    //         );
    //     }

    //     println!("Mul test passed! Result: {:?}", out_f32);
    // }
}
