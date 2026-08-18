//! CUDA integration tests: graph-based execution.
//!
//! Skips gracefully when no CUDA device is available. One shared backend for
//! all tests (NVRTC compilation happens once).

use std::sync::OnceLock;

use feml_core::backend::{Backend, BackendRegistrar, BufferUsage};
use feml_core::graph::GraphBuilder;
use feml_core::{DType, GraphPlan, Shape, TensorId, shape};
use feml_cpu::CpuBackend;
use feml_cuda::CudaRegistrar;

static BACKEND: OnceLock<Option<Box<dyn Backend>>> = OnceLock::new();

fn cuda_backend() -> Option<&'static dyn Backend> {
    BACKEND
        .get_or_init(|| {
            let reg = CudaRegistrar::probe().ok()?;
            let device = reg.device(0).ok()?;
            device.init_backend().ok()
        })
        .as_deref()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; v.len() * 4];
    for (i, x) in v.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
    }
    b
}

fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap())
}

fn read_tensor(backend: &dyn Backend, plan: &GraphPlan, id: TensorId) -> Vec<f32> {
    let t = plan.find(id).unwrap();
    let n = t.layout.shape.len();
    let mut out = vec![0u8; n * 4];
    plan.read_tensor(backend, id, &mut out).unwrap();
    (0..n).map(|i| f32_at(&out, i)).collect()
}

fn write_tensor(backend: &dyn Backend, plan: &GraphPlan, id: TensorId, data: &[f32]) {
    plan.write_tensor(backend, id, &bytes(data)).unwrap();
}

fn mul_graph(n: usize) -> (GraphBuilder, TensorId, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let a = g.input("a", DType::F32, Shape::new(&[n]).unwrap()).unwrap();
    let b = g.input("b", DType::F32, Shape::new(&[n]).unwrap()).unwrap();
    let c = g.mul(a, b).unwrap();
    g.mark_output(c).unwrap();
    (g, a, b, c)
}

fn add_graph(n: usize) -> (GraphBuilder, TensorId, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let a = g.input("a", DType::F32, Shape::new(&[n]).unwrap()).unwrap();
    let b = g.input("b", DType::F32, Shape::new(&[n]).unwrap()).unwrap();
    let c = g.add(a, b).unwrap();
    g.mark_output(c).unwrap();
    (g, a, b, c)
}

fn mul_mat_graph() -> (GraphBuilder, TensorId, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let w = g.weight("w", DType::F32, shape![16, 8]).unwrap();
    let x = g.input("x", DType::F32, shape![16, 32]).unwrap();
    let out = g.mul_mat(w, x).unwrap();
    g.mark_output(out).unwrap();
    (g, w, x, out)
}

#[test]
fn mul_roundtrip_matches_host() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let (g, a, b, c) = mul_graph(1024);
    let plan = g.compile(backend).unwrap();

    let src0: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let src1: Vec<f32> = (0..1024).map(|i| i as f32 * 0.25 + 1.0).collect();
    write_tensor(backend, &plan, a, &src0);
    write_tensor(backend, &plan, b, &src1);

    backend.graph_compute(&plan).unwrap();
    backend.synchronize().unwrap();

    let got = read_tensor(backend, &plan, c);
    for i in 0..1024 {
        let want = src0[i] * src1[i];
        assert!((got[i] - want).abs() < 1e-4, "mul mismatch at {i}: got {}, want {want}", got[i]);
    }
}

#[test]
fn add_roundtrip_matches_host() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let (g, a, b, c) = add_graph(1024);
    let plan = g.compile(backend).unwrap();

    let src0: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let src1: Vec<f32> = (0..1024).map(|i| i as f32 * 0.25 + 1.0).collect();
    write_tensor(backend, &plan, a, &src0);
    write_tensor(backend, &plan, b, &src1);

    backend.graph_compute(&plan).unwrap();
    backend.synchronize().unwrap();

    let got = read_tensor(backend, &plan, c);
    for i in 0..1024 {
        let want = src0[i] + src1[i];
        assert!((got[i] - want).abs() < 1e-4, "add mismatch at {i}: got {}, want {want}", got[i]);
    }
}

#[test]
fn mul_broadcast_matches_host() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    // [3] broadcasts over [2, 3] (ggml trailing-dim rule).
    let mut g = GraphBuilder::new();
    let a = g.input("a", DType::F32, shape![2, 3]).unwrap();
    let b = g.input("b", DType::F32, shape![3]).unwrap();
    let c = g.mul(a, b).unwrap();
    g.mark_output(c).unwrap();
    let plan = g.compile(backend).unwrap();

    let src0: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    let src1: Vec<f32> = vec![1.0, 2.0, 3.0];
    write_tensor(backend, &plan, a, &src0);
    write_tensor(backend, &plan, b, &src1);

    backend.graph_compute(&plan).unwrap();
    backend.synchronize().unwrap();

    let got = read_tensor(backend, &plan, c);
    for i in 0..6 {
        // [3] broadcasts over the trailing dim of [2, 3]: flat index i has
        // trailing-dim coordinate i % 2.
        let want = src0[i] * src1[i % 2];
        assert!(
            (got[i] - want).abs() < 1e-4,
            "broadcast mul mismatch at {i}: got {}, want {want}",
            got[i]
        );
    }
}

#[test]
fn mul_broadcast_src0_smaller_agrees_with_cpu() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let cpu = CpuBackend::new();

    // [3] as src0 broadcasts over [2, 3] src1 -> dst [2, 3] is larger than
    // src0 in dim 1 (regression: the grid must cover dst dims).
    let mut g = GraphBuilder::new();
    let a = g.input("a", DType::F32, shape![3]).unwrap();
    let b = g.input("b", DType::F32, shape![2, 3]).unwrap();
    let c = g.mul(a, b).unwrap();
    g.mark_output(c).unwrap();

    let cpu_plan = g.compile(&cpu).unwrap();
    let gpu_plan = g.compile(backend).unwrap();

    let src0: Vec<f32> = vec![1.0, 2.0, 3.0];
    let src1: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    write_tensor(&cpu, &cpu_plan, a, &src0);
    write_tensor(&cpu, &cpu_plan, b, &src1);
    write_tensor(backend, &gpu_plan, a, &src0);
    write_tensor(backend, &gpu_plan, b, &src1);

    cpu.graph_compute(&cpu_plan).unwrap();
    backend.graph_compute(&gpu_plan).unwrap();
    backend.synchronize().unwrap();

    let cpu_out = read_tensor(&cpu, &cpu_plan, c);
    let gpu_out = read_tensor(backend, &gpu_plan, c);
    assert_eq!(cpu_out.len(), gpu_out.len());
    assert_eq!(cpu_out.len(), 6);
    for (i, (x, y)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        assert!((x - y).abs() < 1e-4, "CPU/CUDA broadcast mismatch at {i}: cpu {x}, gpu {y}");
    }
}

#[test]
fn mul_mat_matches_reference() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let (g, w, x, out) = mul_mat_graph();
    let plan = g.compile(backend).unwrap();

    let w_data: Vec<f32> = (0..16 * 8).map(|i| (i as f32) * 0.1).collect();
    let x_data: Vec<f32> = (0..16 * 32).map(|i| (i as f32) * 0.01 - 0.5).collect();
    write_tensor(backend, &plan, w, &w_data);
    write_tensor(backend, &plan, x, &x_data);

    backend.graph_compute(&plan).unwrap();
    backend.synchronize().unwrap();

    let got = read_tensor(backend, &plan, out);
    for m in 0..8 {
        for n in 0..32 {
            // ggml layout: [K, M] with K contiguous -> w[k, m] at k + 16*m;
            // result [M, N] has M contiguous -> flat index = m + n*8
            let want: f32 = (0..16).map(|k| w_data[k + 16 * m] * x_data[k + 16 * n]).sum();
            let val = got[m + n * 8];
            assert!(
                (val - want).abs() < 1e-3,
                "mul_mat mismatch at [{m},{n}]: got {val}, want {want}"
            );
        }
    }
}

#[test]
fn cpu_and_cuda_agree() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let cpu = CpuBackend::new();

    let (g, w, x, out) = mul_mat_graph();
    let cpu_plan = g.compile(&cpu).unwrap();
    let gpu_plan = g.compile(backend).unwrap();

    let w_data: Vec<f32> = (0..16 * 8).map(|i| i as f32 * 0.3).collect();
    let x_data: Vec<f32> = (0..16 * 32).map(|i| i as f32 * 0.05 - 1.0).collect();
    write_tensor(&cpu, &cpu_plan, w, &w_data);
    write_tensor(&cpu, &cpu_plan, x, &x_data);
    write_tensor(backend, &gpu_plan, w, &w_data);
    write_tensor(backend, &gpu_plan, x, &x_data);

    cpu.graph_compute(&cpu_plan).unwrap();
    backend.graph_compute(&gpu_plan).unwrap();
    backend.synchronize().unwrap();

    let cpu_out = read_tensor(&cpu, &cpu_plan, out);
    let gpu_out = read_tensor(backend, &gpu_plan, out);
    assert_eq!(cpu_out.len(), gpu_out.len());
    for (i, (a, b)) in cpu_out.iter().zip(gpu_out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "CPU/CUDA mismatch at {i}: cpu {a}, gpu {b}");
    }
}

#[test]
fn fill_and_read_roundtrip() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let buf = backend.create_buffer(64, BufferUsage::Compute).unwrap();
    backend.fill(buf, 0, 0xAB, 64).unwrap();
    backend.synchronize().unwrap();
    let mut out = vec![0u8; 64];
    backend.read(buf, 0, &mut out).unwrap();
    assert!(out.iter().all(|&b| b == 0xAB));
}

#[test]
fn out_of_bounds_write_rejected() {
    let backend = match cuda_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no CUDA device");
            return;
        }
    };
    let buf = backend.create_buffer(8, BufferUsage::Compute).unwrap();
    assert!(backend.write(buf, 4, &[1, 2, 3, 4, 5]).is_err());
}

#[test]
fn registrar_reports_devices() {
    let reg = CudaRegistrar::probe().unwrap();
    let _ = reg.device_count();
    assert_eq!(reg.name(), "CUDA");
    if reg.device_count() > 0 {
        let dev = reg.device(0).unwrap();
        let info = dev.info().unwrap();
        assert!(!info.name.is_empty());
        assert!(reg.score() > 0);
    } else {
        assert_eq!(reg.score(), 0);
    }
}
