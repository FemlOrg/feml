# Feml

Feml — High-performance Rust rewrite of GGML, supporting CPU and GPU backends with zero runtime memory allocation.

## Backend Status

| Backend | Feature | Status |
| --- | --- | --- |
| CPU | `cpu` (default) | Registry discovery, backend opening, buffer allocation, tensor binding, fill/read/write/copy, and F32 `TensorOpMul` graph compute. |
| OpenCL | `opencl` | Optional backend with device discovery, buffer allocation, and kernel-backed `TensorOpMul`. Some async and device-info paths are still incomplete. |
| CUDA | `cuda` | Optional backend with device discovery, buffer allocation, and kernel-backed `TensorOpMul`. Some buffer initialization and async paths are still incomplete. |

## Develop

### Build Feml

```shell
cargo build # build feml shared library
```

The default feature set enables the CPU backend. Optional backends can be built with features:

```shell
cargo build --features opencl
cargo build --features cuda
```

### Test

```shell
cargo test
cargo test --test cpu_tests
```

### Format Code

```shell
cargo fmt   # format project

cargo fmt --check # check format is right (not modify files)

rustfmt src/*.rs # format project
```
