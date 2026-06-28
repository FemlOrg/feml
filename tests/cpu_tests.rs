#[cfg(feature = "cpu")]
mod cpu_backend {
    use feml::backend::BackendBufferUsage;
    use feml::compute_graph::ComputeGraph;
    use feml::context::Context;
    use feml::data_type::{DataType, TensorOpType, TensorType};
    use feml::registry::Registry;
    use feml::shape;

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn registry_cpu_open_backend() {
        let registry = Registry::discover().expect("registry discover should succeed");
        registry.init_all().expect("registry init_all should succeed");

        let cpu = registry.find("CPU").expect("CPU backend register should exist");
        assert_eq!(cpu.device_count(), 1);
        assert!(registry.open_backend("CPU", 0).is_ok());
    }

    #[test]
    fn backend_buffer_write_read() {
        let registry = Registry::discover().expect("registry discover should succeed");
        registry.init_all().expect("registry init_all should succeed");
        let backend = registry.open_backend("CPU", 0).expect("CPU backend should open");
        let buffer = backend
            .create_buffer(64, BackendBufferUsage::Any)
            .expect("CPU buffer should be created");

        let mut ctx = Context::builder().tensor_pool_capacity(4).build();
        let tensor = ctx
            .new_tensor(DataType::F32, &shape![4, 1, 1, 1])
            .expect("tensor should be created");
        buffer.init_tensor(tensor.clone(), 0).expect("tensor should bind to CPU buffer");

        let mut input = encode_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mut output = vec![0; input.len()];
        let size = input.len();
        buffer.write(tensor.clone(), &mut input, 0, size).expect("write should succeed");
        buffer.read(tensor, &mut output, 0, size).expect("read should succeed");

        assert_eq!(decode_f32(&output), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn graph_compute_mul_f32() {
        let registry = Registry::discover().expect("registry discover should succeed");
        registry.init_all().expect("registry init_all should succeed");
        let backend = registry.open_backend("CPU", 0).expect("CPU backend should open");
        let buffer = backend
            .create_buffer(128, BackendBufferUsage::Any)
            .expect("CPU buffer should be created");

        let mut ctx = Context::builder().tensor_pool_capacity(8).build();
        let shape = shape![4, 1, 1, 1];
        let mut lhs = ctx.new_tensor(DataType::F32, &shape).unwrap();
        let rhs = ctx.new_tensor(DataType::F32, &shape).unwrap();

        lhs.set_tensor_type(TensorType::FlagParam);
        lhs.set_op_type(TensorOpType::TensorNone);
        rhs.set_tensor_type(TensorType::FlagParam);
        rhs.set_op_type(TensorOpType::TensorNone);

        let dst = lhs.mul(rhs.clone()).expect("mul tensor should be created");

        buffer.init_tensor(lhs.clone(), 0).unwrap();
        buffer.init_tensor(rhs.clone(), 32).unwrap();
        buffer.init_tensor(dst.clone(), 64).unwrap();

        let mut lhs_bytes = encode_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mut rhs_bytes = encode_f32(&[10.0, 20.0, 30.0, 40.0]);
        let lhs_size = lhs_bytes.len();
        let rhs_size = rhs_bytes.len();
        buffer.write(lhs.clone(), &mut lhs_bytes, 0, lhs_size).unwrap();
        buffer.write(rhs.clone(), &mut rhs_bytes, 0, rhs_size).unwrap();

        let mut graph = ComputeGraph::new();
        graph.build_forward(&ctx, dst.tensor_id(), false).unwrap();
        backend.graph_compute(&ctx, &mut graph).expect("CPU graph compute should succeed");

        let mut output = vec![0; dst.nbytes()];
        let output_size = output.len();
        buffer.read(dst, &mut output, 0, output_size).unwrap();

        assert_eq!(decode_f32(&output[..16]), vec![10.0, 40.0, 90.0, 160.0]);
    }
}
