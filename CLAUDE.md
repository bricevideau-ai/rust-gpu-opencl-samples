# rust-gpu OpenCL Samples

## Overview

OpenCL compute kernel samples written in Rust using rust-gpu. Kernels compile to OpenCL SPIR-V and run on any OpenCL device with SPIR-V IL support (pocl 6.0+, Intel GPU/CPU, etc.).

Depends on the `opencl-kernel-support` branch of https://github.com/bricevideau-ai/rust-gpu

## Project structure

```
kernels/
  collatz/      — Collatz conjecture kernel
  mandelbrot/   — Mandelbrot set with complex numbers (num-complex)
runner/         — Host-side OpenCL runner with helpers
```

## Key patterns

### Kernel crates
- `#![cfg_attr(target_arch = "spirv", no_std)]` at the top
- `crate-type = ["dylib"]` in Cargo.toml
- Entry points: `#[spirv(kernel)]`
- Global buffers: `#[spirv(cross_workgroup)] data: &mut [u32]`
- Work item ID: `#[spirv(global_invocation_id)] id: USizeVec3`
- Scalar args: just by value (`max_iter: u32`)
- Struct args: by value, works for scalar-only structs (`vp: Viewport`)

### Runner
- `OclContext` — wraps device, context, queue
- `DeviceSlice<T>` — buffer + length pair (matches Rust-GPU slice decomposition)
- `KernelArg` trait — implemented for `DeviceSlice`, `u32`, `f32`, `Viewport`
- Slices automatically set two kernel args (pointer + usize length)

### Dependencies
- Uses `use-compiled-tools` feature (not `use-installed-tools`) — spirv-opt runs in-process with crash isolation via fork()
- `num-complex` with `default-features = false` works on SPIR-V targets
- `glam >= 0.30.8` with `default-features = false` for vector types

## How to build and run

```bash
cargo check -p collatz -p mandelbrot -p runner   # check everything compiles
cargo run -p runner --release                     # run all samples (needs OpenCL runtime)
cargo run -p runner --release -- collatz           # run specific sample
cargo run -p runner --release -- mandelbrot        # run specific sample
```

## How to add a new sample

1. Create `kernels/mykernel/Cargo.toml` with `crate-type = ["dylib"]` and `spirv-std`/`glam` deps
2. Write the kernel in `kernels/mykernel/src/lib.rs` with `#[spirv(kernel)]` entry point
3. Add `"kernels/mykernel"` to workspace members in root `Cargo.toml`
4. Add a `run_mykernel` function in `runner/src/main.rs` following the existing pattern
5. Add the `KernelArg` impl for any custom struct types
6. Wire it into the `main()` match

## Conventions

- `cargo fmt --all` before committing
- `cargo clippy --all-targets -- -D warnings` must pass
- Kernel crates inherit workspace lints (`[lints] workspace = true`)
- The workspace has `check-cfg` for `target_arch = "spirv"` to suppress clippy warnings on kernel crates
- CI is build-only (no runtime execution until Ubuntu 26.04 runners have pocl with SPIR-V)

## Known issues

- `is_multiple_of(2)` crashes spirv-opt's `DeadBranchElimPass`. The compiled-tools path isolates this via fork() and falls back to safe optimization passes. Use `% 2 == 0` if using `use-installed-tools`.
- CI cannot run kernels — Ubuntu 24.04's pocl lacks SPIR-V IL support.
- Verified working on: pocl 6.0 (CPU), Intel Gen11 GPU, Intel CPU (pocl).
