//! OpenCL integration tests: graph-based execution.
//!
//! Skips gracefully when no OpenCL device is available. One shared backend
//! for all tests: some OpenCL drivers (e.g. Glenfly) segfault on concurrent
//! context creation from parallel test threads.

use feml_core::backend::{Backend, BackendRegistrar, BufferUsage};
use feml_core::graph::GraphBuilder;
use feml_core::{DType, GraphPlan, Shape, TensorId, shape};
use feml_cpu::CpuBackend;
use feml_opencl::OpenclRegistrar;
use std::sync::OnceLock;

static BACKEND: OnceLock<Option<Box<dyn Backend>>> = OnceLock::new();

fn opencl_backend() -> Option<&'static dyn Backend> {
    BACKEND
        .get_or_init(|| {
            let reg = OpenclRegistrar::probe().ok()?;
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
    let backend = match opencl_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no OpenCL device");
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
fn mul_mat_matches_reference() {
    let backend = match opencl_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no OpenCL device");
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
fn cpu_and_opencl_agree() {
    let backend = match opencl_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no OpenCL device");
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
        assert!((a - b).abs() < 1e-3, "CPU/OpenCL mismatch at {i}: cpu {a}, gpu {b}");
    }
}

#[test]
fn fill_and_read_roundtrip() {
    let backend = match opencl_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no OpenCL device");
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
    let backend = match opencl_backend() {
        Some(b) => b,
        None => {
            eprintln!("skipping: no OpenCL device");
            return;
        }
    };
    let buf = backend.create_buffer(8, BufferUsage::Compute).unwrap();
    assert!(backend.write(buf, 4, &[1, 2, 3, 4, 5]).is_err());
}

#[test]
fn registrar_reports_devices() {
    let reg = OpenclRegistrar::probe().unwrap();
    let _ = reg.device_count();
    assert_eq!(reg.name(), "OpenCL");
    if reg.device_count() > 0 {
        let dev = reg.device(0).unwrap();
        let info = dev.info().unwrap();
        assert!(!info.name.is_empty());
        assert!(reg.score() > 0);
    } else {
        assert_eq!(reg.score(), 0);
    }
}
