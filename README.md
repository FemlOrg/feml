# Feml

Feml — a high-performance Rust operator library for LLM inference (a Rust rewrite of GGML).

It inherits ggml's core strengths (static compute graphs, graph-aware memory
planning, multiple backends, quantized types) while fixing its inherent
weaknesses with Rust's type safety and ecosystem (memory safety, error
propagation, package distribution).

## Core features

- **Three-phase API**: `GraphBuilder → compile → backend.graph_compute(plan)`, with zero runtime memory allocation (the CPU path is locked by a counting-allocator test)
- **Zero-allocation execution**: `MemoryPlanner` (a port of ggml-alloc) performs liveness analysis, inplace reuse and view sharing at compile time, producing an immutable `GraphPlan`
- **Multiple backends**: `feml-cpu` / `feml-cuda` / `feml-opencl` behind a unified `Backend` trait (`Send + Sync`, zero panics, out-of-bounds is an error)
- **ggml-compatible data layout**: `[K, M]` contiguous, quantized type table (Q4_0 ~ Q8_K) mapped 1:1 to GGML
- **Inference op interfaces**: mul / add / mul_mat / rms_norm / silu / softmax / rope / get_rows / concat / copy / mul_mat_id / diag_mask_inf / scale

## Repository layout

```
feml/                    workspace
├── feml-core/           tensor types, graph, plan, memory planner, backend traits (zero dependencies)
├── feml-cpu/            CPU backend (mul/add/mul_mat kernels)
├── feml-cuda/           CUDA backend (cudarc; mul/add/mul_mat kernels, NVRTC at runtime)
├── feml-opencl/         OpenCL backend (strided mul + mul_mat kernels)
├── legacy/              v0 code (kept in git history, not a workspace member)
└── docs/                DESIGN.md / backend-guide.md / api-guide.md
```

## Quick start

```rust
use feml_core::prelude::*;
use feml_cpu::CpuBackend;

let backend = CpuBackend::new();
let mut g = GraphBuilder::new();

let a = g.input("a", DType::F32, shape![16, 8])?;
let b = g.input("b", DType::F32, shape![16, 8])?;
let c = g.mul(a, b)?;
g.mark_output(c)?;

let plan = g.compile(&backend)?;      // topo sort + memory planning + buffer allocation
backend.graph_compute(&plan)?;        // zero-allocation execution
```

See [docs/api-guide.md](docs/api-guide.md) for detailed usage.

## Build & test

```shell
cargo build --workspace
cargo test --workspace          # 83 tests (numeric verification on a real CUDA GPU; OpenCL/CUDA tests skip without a device)
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

## Design docs

- [docs/DESIGN.md](docs/DESIGN.md) — overall architecture and roadmap (M0–M5)
- [docs/backend-guide.md](docs/backend-guide.md) — third-party backend integration guide
- [docs/api-guide.md](docs/api-guide.md) — API usage guide

## License

MIT
