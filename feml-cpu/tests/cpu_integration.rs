//! CPU backend integration tests.

use feml_core::backend::{Backend, BackendRegistrar, BufferUsage};
use feml_core::graph::GraphBuilder;
use feml_core::op::Op;
use feml_core::{DType, GraphPlan, Shape, TensorId, shape};
use feml_cpu::CpuBackend;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// ---- counting allocator: verifies zero-allocation execution --------------

static ALLOC_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;

thread_local! {
    static COUNTING_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING_ON.with(|c| c.get()) {
            ALLOC_COUNTER.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

// ---- helpers --------------------------------------------------------------

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

fn read_tensor(backend: &CpuBackend, plan: &GraphPlan, id: TensorId) -> Vec<f32> {
    let t = plan.find(id).unwrap();
    let n = t.layout.shape.len();
    let mut out = vec![0u8; n * 4];
    plan.read_tensor(backend, id, &mut out).unwrap();
    (0..n).map(|i| f32_at(&out, i)).collect()
}

fn mul_graph(n: usize) -> (GraphBuilder, TensorId, TensorId, TensorId) {
    let mut g = GraphBuilder::new();
    let a = g.input("a", DType::F32, Shape::new(&[n]).unwrap()).unwrap();
    let b = g.input("b", DType::F32, Shape::new(&[n]).unwrap()).unwrap();
    let c = g.mul(a, b).unwrap();
    g.mark_output(c).unwrap();
    (g, a, b, c)
}

// ---- tests ----------------------------------------------------------------

#[test]
fn graph_mul_roundtrip() {
    let backend = CpuBackend::new();
    let (g, a, b, c) = mul_graph(1024);
    let plan = g.compile(&backend).unwrap();

    let src0: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let src1: Vec<f32> = (0..1024).map(|i| i as f32 * 0.25 + 1.0).collect();

    plan.write_tensor(&backend, a, &bytes(&src0)).unwrap();
    plan.write_tensor(&backend, b, &bytes(&src1)).unwrap();

    backend.graph_compute(&plan).unwrap();

    let got = read_tensor(&backend, &plan, c);
    for i in 0..1024 {
        let want = src0[i] * src1[i];
        assert!((got[i] - want).abs() < 1e-4, "mul mismatch at {i}: got {}, want {want}", got[i]);
    }
}

#[test]
fn graph_add_roundtrip() {
    let backend = CpuBackend::new();
    let mut g = GraphBuilder::new();
    let a = g.input("a", DType::F32, shape![8]).unwrap();
    let b = g.input("b", DType::F32, shape![8]).unwrap();
    let c = g.add(a, b).unwrap();
    g.mark_output(c).unwrap();
    let plan = g.compile(&backend).unwrap();

    plan.write_tensor(&backend, a, &bytes(&[1.0; 8])).unwrap();
    plan.write_tensor(&backend, b, &bytes(&[2.0; 8])).unwrap();
    backend.graph_compute(&plan).unwrap();

    let got = read_tensor(&backend, &plan, c);
    assert!(got.iter().all(|&x| (x - 3.0).abs() < 1e-6));
}

#[test]
fn fill_and_read_roundtrip() {
    let backend = CpuBackend::new();
    let buf = backend.create_buffer(64, BufferUsage::Compute).unwrap();
    backend.fill(buf, 0, 0xAB, 64).unwrap();
    let mut out = vec![0u8; 64];
    backend.read(buf, 0, &mut out).unwrap();
    assert!(out.iter().all(|&b| b == 0xAB));
}

#[test]
fn write_read_partial_offset() {
    let backend = CpuBackend::new();
    let buf = backend.create_buffer(16, BufferUsage::Compute).unwrap();
    backend.fill(buf, 0, 0x00, 16).unwrap();
    backend.write(buf, 8, &[1, 2, 3]).unwrap();
    let mut out = vec![0u8; 16];
    backend.read(buf, 0, &mut out).unwrap();
    assert_eq!(&out[0..8], &[0u8; 8]);
    assert_eq!(&out[8..11], &[1, 2, 3]);
    assert_eq!(&out[11..16], &[0u8; 5]);
}

#[test]
fn out_of_bounds_write_rejected() {
    let backend = CpuBackend::new();
    let buf = backend.create_buffer(8, BufferUsage::Compute).unwrap();
    assert!(backend.write(buf, 4, &[1, 2, 3, 4, 5]).is_err());
    assert!(backend.read(buf, 4, &mut [0u8; 5]).is_err());
}

#[test]
fn unknown_handle_rejected() {
    let backend = CpuBackend::new();
    let ghost = feml_core::backend::BufferHandle::new(999);
    assert!(backend.read(ghost, 0, &mut [0u8; 1]).is_err());
}

#[test]
fn registrar_reports_cpu_device() {
    let reg = feml_cpu::CpuRegistrar::probe().unwrap();
    assert_eq!(reg.name(), "CPU");
    assert_eq!(reg.device_count(), 1);
    assert!(reg.score() > 0);
    let dev = reg.device(0).unwrap();
    let info = dev.info().unwrap();
    assert_eq!(info.name, "CPU");
    assert!(dev.supports_op(Op::Mul, &[DType::F32]));
}

#[test]
fn registry_open_best_picks_cpu() {
    let mut reg = feml_core::Registry::new();
    reg.register(Box::new(feml_cpu::CpuRegistrar::probe().unwrap()));
    let backend = reg.open_best().unwrap();
    assert_eq!(backend.name(), "CPU");
}

#[test]
fn supports_op_rejects_unsupported() {
    let backend = CpuBackend::new();
    assert!(backend.supports_op(Op::Mul, &[DType::F32]));
    assert!(!backend.supports_op(Op::Mul, &[DType::Q4_0]));
    assert!(!backend.supports_op(Op::RmsNorm { eps: 1e-5 }, &[DType::F32]));
}

#[test]
fn mul_mat_matches_reference() {
    let backend = CpuBackend::new();
    let mut g = GraphBuilder::new();
    let w = g.weight("w", DType::F32, shape![16, 8]).unwrap();
    let x = g.input("x", DType::F32, shape![16, 32]).unwrap();
    let out = g.mul_mat(w, x).unwrap();
    g.mark_output(out).unwrap();
    let plan = g.compile(&backend).unwrap();

    let w_data: Vec<f32> = (0..16 * 8).map(|i| (i as f32) * 0.1).collect();
    let x_data: Vec<f32> = (0..16 * 32).map(|i| (i as f32) * 0.01 - 0.5).collect();
    plan.write_tensor(&backend, w, &bytes(&w_data)).unwrap();
    plan.write_tensor(&backend, x, &bytes(&x_data)).unwrap();

    backend.graph_compute(&plan).unwrap();

    let got = read_tensor(&backend, &plan, out);
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
fn plan_release_frees_buffers() {
    let backend = CpuBackend::new();
    let (g, _, _, c) = mul_graph(4);
    let plan = g.compile(&backend).unwrap();
    let handle = plan.buffers[plan.find(c).unwrap().buffer].handle;
    plan.release(&backend).unwrap();
    assert!(
        backend.read(handle, 0, &mut [0u8; 4]).is_err(),
        "handle must be invalid after plan release"
    );
}

#[test]
fn tensor_io_validates_size() {
    let backend = CpuBackend::new();
    let (g, a, _, _) = mul_graph(4);
    let plan = g.compile(&backend).unwrap();
    assert!(plan.write_tensor(&backend, a, &[0u8; 7]).is_err(), "wrong size must be rejected");
    assert!(plan.read_tensor(&backend, a, &mut [0u8; 9]).is_err());
    let ghost = TensorId::new();
    assert!(plan.write_tensor(&backend, ghost, &[0u8; 16]).is_err());
}

#[test]
fn execute_is_zero_allocation() {
    let backend = CpuBackend::new();
    let (g, a, b, c) = mul_graph(4096);
    let plan = g.compile(&backend).unwrap();

    let src0: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    let src1: Vec<f32> = (0..4096).map(|i| 1.0 - i as f32 * 0.001).collect();
    plan.write_tensor(&backend, a, &bytes(&src0)).unwrap();
    plan.write_tensor(&backend, b, &bytes(&src1)).unwrap();

    backend.graph_compute(&plan).unwrap();
    let _ = read_tensor(&backend, &plan, c);

    ALLOC_COUNTER.store(0, Ordering::Relaxed);
    COUNTING_ON.with(|f| f.set(true));
    backend.graph_compute(&plan).unwrap();
    COUNTING_ON.with(|f| f.set(false));

    let allocs = ALLOC_COUNTER.load(Ordering::Relaxed);
    assert_eq!(allocs, 0, "plan.execute must not allocate");
}
