# rust-gpu OpenCL Samples

## Overview

OpenCL compute kernel samples written in Rust using rust-gpu. Kernels compile to OpenCL SPIR-V and run on any OpenCL device with SPIR-V IL support (pocl 6.0+, Intel GPU/CPU, etc.).

Depends on the `opencl-kernel-support` branch of https://github.com/bricevideau-ai/rust-gpu

## Project structure

```
kernels/
  collatz/      — Collatz conjecture kernel
  mandelbrot/   — Mandelbrot set with complex numbers (num-complex), uses printf
  reduce/       — Hierarchical reduction using subgroup ops + shared memory (OpenCL 2.0)
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

### OpenCL 2.0 / subgroup kernels
- Target `spirv-unknown-opencl2.0` with `.capability(Capability::Groups)` in SpirvBuilder
- `#[spirv(kernel(threads(N)))]` to declare workgroup size
- `#[spirv(workgroup)] shared: &mut [T; N]` for local/shared memory (module-scope, not a kernel arg)
- `#[spirv(subgroup_id)]`, `#[spirv(subgroup_local_invocation_id)]`, `#[spirv(num_subgroups)]` builtins
- `workgroup_memory_barrier_with_group_sync()` for barriers
- `group_exclusive_i_add`, `group_i_add`, etc. from `spirv_std::arch` for subgroup ops

### Printf
- `spirv_std::printf!("format %u\n", value)` — OpenCL printf via `OpenCL.std` extended instruction set
- `spirv_std::printfln!("format %u", value)` — auto-appends newline
- `%f` accepts both f32 and f64 (no promotion needed)
- Requires `#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]` in the kernel crate

### Runner
- `OclContext` — wraps device, context, queue
- `DeviceSlice<T>` — buffer + length pair (matches Rust-GPU slice decomposition)
- `KernelArg` trait — implemented for `DeviceSlice`, `u32`, `f32`, `Viewport`
- Slices automatically set two kernel args (pointer + usize length)
- `run()` accepts optional `local_work_size` for workgroup-based kernels
- `compile_kernel()` for OpenCL 1.2, `compile_kernel_opencl2()` for OpenCL 2.0 + Groups

### Dependencies
- Uses `use-compiled-tools` feature — spirv-opt runs in-process with crash isolation via fork()
- `num-complex` with `default-features = false` works on SPIR-V targets
- `glam >= 0.30.8` with `default-features = false` for vector types

## How to build and run

```bash
cargo check -p collatz -p mandelbrot -p reduce -p runner  # check everything compiles
cargo run -p runner --release                              # run all samples (needs OpenCL runtime)
cargo run -p runner --release -- collatz                   # run specific sample
cargo run -p runner --release -- mandelbrot                # run specific sample
cargo run -p runner --release -- reduce                    # run specific sample (OpenCL 2.0)
cargo run -p runner --release -- debug-abort               # debug-printf abort strategy demo (not in default set)
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
- CI is build-only (no runtime execution until Ubuntu runners have pocl with SPIR-V)

## Known issues

- `is_multiple_of(2)` crashes spirv-opt's `DeadBranchElimPass`. The compiled-tools path isolates this via fork() and falls back to safe optimization passes.
- pocl 7.0+ has a regression with multi-workgroup kernels using subgroup ops + subgroup builtins + shared memory (last workgroup gets wrong data). Works correctly on pocl 6.0 and Intel GPU.
- Verified working on: pocl 6.0 (CPU), pocl 7.2-pre (CPU, with caveats above), Intel Gen11 GPU.
