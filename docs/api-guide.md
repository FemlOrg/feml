# feml API Guide

This guide covers the full usage of `feml-core`. Crate responsibilities:
`feml-core` (tensors / graphs / plans / backend traits), `feml-cpu`, `feml-opencl`
(concrete backends).

## 1. Core concepts

| Concept | Type | Description |
|---|---|---|
| Tensor handle | `TensorId` (Copy) | Build-phase reference to a tensor; not a data pointer |
| Metadata | `TensorMeta` | Build-phase tensor description (dtype/shape/op/srcs/flags) |
| Graph builder | `GraphBuilder` | Build-phase API; all validation returns `Result` |
| Compiled artifact | `GraphPlan` | Immutable, `Send`; handles and offsets fully resolved |
| Execution stream | `dyn Backend` | Buffer lifecycle + `graph_compute` |
| Backend registry | `Registry` | Backend discovery and selection (`open_best`) |

**Three-phase flow** (the core design):

```
GraphBuilder (build, allocates freely) → compile(&backend) → GraphPlan (immutable)
                                                            → backend.graph_compute(&plan)  // zero-allocation
```

## 2. Minimal complete example

```rust
use feml_core::prelude::*;
use feml_cpu::CpuBackend;

fn main() -> Result<()> {
    let backend = CpuBackend::new();                 // 1. pick a backend

    let mut g = GraphBuilder::new();                 // 2. build the graph
    let a = g.input("a", DType::F32, shape![16, 8])?;
    let b = g.input("b", DType::F32, shape![16, 8])?;
    let c = g.mul(a, b)?;
    g.mark_output(c)?;

    let plan = g.compile(&backend)?;                 // 3. compile (validate + plan + allocate)
    backend.graph_compute(&plan)?;                   // 4. zero-allocation execution

    // 5. read/write data (buffer-level API + plan offsets)
    let t = plan.find(c).unwrap();
    backend.write(plan.buffers[t.buffer].handle, t.offset, &data_bytes)?;
    let mut out = vec![0u8; t.nbytes()];
    backend.read(plan.buffers[t.buffer].handle, t.offset, &mut out)?;

    plan.release(&backend)?;                         // 6. (optional) free the plan's buffers
    Ok(())
}
```

## 3. Creating tensors

```rust
let x = g.input("x", DType::F32, shape![4096])?;         // written by the user before every run
let w = g.weight("w", DType::F32, shape![4096, 11008])?; // resident weight (Weights buffer, never reused)
let v = g.view(w, 0, shape![2048, 11008])?;              // zero-copy view (offset must be 4-byte aligned)
```

`TensorFlags` (set via `mark_output` for `OUTPUT`):
`INPUT` (written each run) / `WEIGHT` (resident) / `OUTPUT` (never freed, user reads it).

## 4. Op methods (full current interface)

| Method | Shape semantics | Validation |
|---|---|---|
| `mul(a, b)` / `add(a, b)` | broadcast (ggml rules, trailing dims align) | matching dtypes, broadcastable |
| `mul_mat(w, x)` | `[K,M] × [K,N] → [M,N]` (M contiguous) | equal inner dim, rank ≤ 2, f32 |
| `rms_norm(x, eps)` | same shape | — |
| `silu(x)` | same shape | — |
| `softmax(x, axis)` | same shape | `axis < 0` means the last dim |
| `rope(x, pos, RopeParams)` | same shape | `pos` must be I32 |
| `get_rows(table, idx)` | `[E, D] × [N] → [E, N]` | `idx` must be I32 |
| `concat(a, b, axis)` | join along `axis` | equal rank, off-axis dims equal, axis in range |
| `copy(src)` | same shape | — |
| `mul_mat_id(w, x, ids, n_sel)` | `[K,M,E] × [K,N] × [E,N] → [M,N]` | `ids` I32, expert count matches |
| `diag_mask_inf(x, n_past)` | same shape | — |
| `scale(x, alpha)` | same shape | — |

**Note**: M2 shipped only the interfaces above (shape inference + validation).
Kernel status: `mul` / `add` / `mul_mat` are implemented on CPU/OpenCL; graphs
using the other ops fail at `compile` (fail-fast "backend does not support op").
Add kernels per `docs/backend-guide.md`.

## 5. Compile behavior (fail-fast)

`compile` runs, in order:
1. validates at least one output is marked;
2. topological sort (iterative DFS, no recursion);
3. **checks every node's op against `Backend::supports_op`** — an unsupported
   op fails immediately;
4. `MemoryPlanner` assigns `(buffer, offset)` per tensor (liveness + inplace
   reuse + view sharing);
5. allocates one big buffer per `Weights` / `Compute` class.

```rust
let err = g.compile(&backend).unwrap_err();
assert!(err.to_string().contains("does not support op softmax"));
```

## 6. Memory model and zero-allocation

- **Compile time**: all metadata and data allocations happen (`plan.tensors` /
  `plan.buffers` are frozen);
- **Execute time**: `graph_compute` only reads/writes at precomputed offsets —
  zero allocation (the CPU path is locked by the counting-allocator test
  `execute_is_zero_allocation`);
- **Buffer ownership**: created by the plan, freed by the plan.
  `plan.release(&backend)` returns them explicitly; the plan is unusable after;
- **Inplace semantics**: for inplace-capable ops (mul/add/rms_norm/silu/
  softmax/rope/diag_mask/scale) `dst` may share memory with `src0` — kernels
  must handle `dst == src0` (read before write).

## 7. Backend selection

```rust
let mut reg = Registry::new();
reg.register(Box::new(CpuRegistrar::probe()?));
reg.register(Box::new(OpenclRegistrar::probe()?));
let backend = reg.open_best()?;   // highest score (GPU > CPU)
```

- `open_best`: highest score among available backends;
- `open("CPU", 0)` / `open("OpenCL", 1)`: by name + device index;
- score convention: CPU = 100, general GPU = 1000, 0 = unavailable.

## 8. Error handling

Every fallible API returns `Result<T, feml_core::Error>` (zero panics).

```rust
Err(Error::shape("mul_mat: inner dim mismatch: 16 vs 32"))   // shape errors
    .context("in graph build")
```

- Build/compile errors carry a context chain;
- Execute errors (out of bounds / unknown handle / unimplemented op) also
  return `Err`, never panic.

## 9. Multithreading

- `GraphPlan` is `Send + Sync`: shareable across threads;
- `Backend` implementations must be `Send + Sync`;
- `graph_compute` currently executes single-threaded (threadpool partitioning
  is an M4 goal); the same plan can be executed serially on multiple threads
  without interference.

## 10. Testing

```shell
cargo test -p feml-core     # unit tests (dtype/layout/planner/graph/registry)
cargo test -p feml-cpu      # CPU integration (mul/mul_mat numerics, zero-alloc, plan release)
cargo test -p feml-opencl   # OpenCL integration (skipped automatically without a device)
```
