# Feml

Feml — High-performance Rust rewrite of GGML, supporting CPU and GPU backends with zero runtime memory allocation.

# Develop

## Build Feml

```shell
cargo build # build feml shared library
```

## Format Code

```shell
cargo fmt   # format project

cargo fmt --check # check format is right (not modify files)

rustfmt src/*.rs # format project
```