# rust-gpu OpenCL Samples

## Overview

OpenCL compute kernel samples written in Rust using rust-gpu. Kernels compile to OpenCL SPIR-V and run on any OpenCL device with SPIR-V IL support (pocl 6.0+, Intel GPU/CPU, etc.).

Depends on the [`opencl-kernel-support`](https://github.com/bricevideau-ai/rust-gpu/tree/opencl-kernel-support) branch of `bricevideau-ai/rust-gpu`. Both `Cargo.toml` and `Cargo.lock` pin against that branch; bump with `cargo update -p spirv-builder -p spirv-std` after a stable promotion.

## Project structure

```
kernels/
  collatz/           — Collatz conjecture kernel
  mandelbrot/        — Mandelbrot set with complex numbers (num-complex), uses printf
  mandelbrot-image/  — Mandelbrot rendered to an OpenCL 2D image (PPM out)
  reduce/            — Hierarchical reduction using subgroup ops + shared memory (OpenCL 2.0)
  raymarch/          — SDF ray marcher built end-to-end on spirv_std::cl::Float3 +
                       spirv_std::arch::opencl_std math intrinsics
  nbody/             — Direct O(N²) gravitational sim built on spirv_std::cl::Double3
                       (OpenCL `double3` ABI), leapfrog integrator
runner/              — Host-side OpenCL runner with helpers
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
- Slice-of-struct args: `&mut [Body]` works as a `cross_workgroup` arg even when the struct contains native `cl::*` vector fields — the host side passes `(ptr, len)` matching the rust-gpu slice decomposition

### Native `cl::*` vector types
Used by `raymarch` (`cl::Float3`, `cl::Int2`) and `nbody` (`cl::Double3`).
- `cl::Float3 / Double3 / IntN / etc.` from `spirv_std::cl::*` — distinct from glam, lowers to native `OpTypeVector` (no per-component scalarisation through glam's scalar fallback) and matches the OpenCL `floatN`/`doubleN` ABI for host interop
- Operators `+ - * /` and `Mul<Scalar>` emit `OpFAdd`/`OpVectorTimesScalar`/etc. directly
- `spirv_std::arch::opencl_std::sqrt(v)` / `length(v)` / `normalize(v)` / `dot(a,b)` etc. accept these via the same `*OrVector` traits as glam — the same line works on both
- Method-form (`a.dot(b)`, `v.normalize()`, `v.length()`) is also wired up via `cl::*` inherent impls; pick free-function vs method by readability
- Host-side fallbacks are provided for every operator and op, so `runner` can compute the same result on CPU for bit-for-bit smoke tests (see the raymarch one-pixel host check)

### Image kernels
- Target `spirv-unknown-opencl1.2` — no explicit capability needed; codegen auto-adds `ImageBasic` when it sees an Image kernel parameter (OpenCL 1.2 supports separate read_only / write_only image kernel args; the same image object can be read in one kernel and written by another)
- For read+write of the same image in a single kernel, use `spirv-unknown-opencl2.0` and explicitly add `.capability(Capability::ImageReadWrite)` (the codegen can't infer this — it changes WriteOnly to ReadWrite)
- `image: &Image!(2D, type=u32, sampled=false)` → AccessQualifier::ReadOnly
- `image: &mut Image!(2D, type=f32, sampled=false)` → AccessQualifier::WriteOnly on 1.2, AccessQualifier::ReadWrite when ImageReadWrite is enabled
- `unsafe { image.write(coord, color) }` / `unsafe { image.read(coord) }` for image I/O
- Host-side: `opencl3::memory::Image::create()` with `cl_image_format` and `cl_image_desc`; pass the image via `image.get()` (returns `cl_mem`) when calling `set_arg`, not `&image`
- Check `device.image_support()` before running image kernels

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

### Debug-printf abort
- Set `.print_metadata(MetadataPrintout::None)` and `.shader_panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit { ... })` on the SpirvBuilder
- Panics in the kernel become `NonSemantic.DebugPrintf` from SPIR-T, which a post-link pass in rust-gpu rewrites to `OpenCL.std` printf — visible on stock OpenCL runtimes that don't support NonSemantic extensions
- Sample: `cargo run -p runner --release -- debug-abort`

### Runner
- `OclContext` — wraps device, context, queue
- `DeviceSlice<T>` — buffer + length pair (matches Rust-GPU slice decomposition)
- `KernelArg` trait — implemented for `DeviceSlice`, `u32`, `f32`, `Viewport`, `cl::*` scalar wrappers used by samples
- Slices automatically set two kernel args (pointer + usize length)
- `run()` accepts optional `local_work_size` for workgroup-based kernels
- `compile_kernel()` for OpenCL 1.2, `compile_kernel_opencl2()` for OpenCL 2.0 + Groups

### Dependencies
- Uses `use-compiled-tools` feature — spirv-opt runs in-process with crash isolation via fork()
- `num-complex` with `default-features = false` works on SPIR-V targets
- `glam >= 0.30.8` with `default-features = false` for vector types

## How to build and run

```bash
cargo check --workspace                                    # check everything compiles
cargo run -p runner --release                              # run all samples (needs OpenCL runtime)
cargo run -p runner --release -- collatz                   # specific samples:
cargo run -p runner --release -- mandelbrot
cargo run -p runner --release -- reduce                    #   (OpenCL 2.0 + Groups)
cargo run -p runner --release -- mandelbrot-image          #   (needs image support)
cargo run -p runner --release -- raymarch                  #   (cl::Float3 + opencl_std math)
cargo run -p runner --release -- nbody                     #   (cl::Double3, needs Float64)
cargo run -p runner --release -- debug-abort               #   (debug-printf abort demo, opt-in)
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

- **`is_multiple_of(2)` crashes spirv-opt's `DeadBranchElimPass`**. The compiled-tools path isolates this via fork() and falls back to safe optimization passes. Upstream: https://github.com/KhronosGroup/SPIRV-Tools/issues/6632
- **pocl `__alignof__` cap on aarch64**: pocl's OpenCL C compiler reports the alignment of every vector type larger than 16 bytes as 16 (so `double3`/`double4` come back 16 instead of 32, `long16` comes back 16 instead of 128, etc.). Affects host/device ABI for OpenCL `*N` vector types. Reproducer: `pocl-vector-align-repro/` (sibling repo). Filed as pocl#2174; fixed by a header-only patch (`__attribute__((aligned(N)))` on every vector typedef in `include/opencl-c-base.h`). Without that patch, host-side `cl_double3` buffers and device-side `cl::Double3` arrays disagree on layout — the rust-gpu codegen and the CPU host both produce the spec-correct values, only pocl's OpenCL C path is off.
- **pocl 7.x multi-workgroup subgroup regression**: a corner case in pocl 7.x with multi-workgroup kernels using subgroup ops + subgroup builtins + shared memory makes the last workgroup get wrong data. Works correctly on pocl 6.0 and Intel GPU.
- **`raymarch` on pocl 6.0**: crashes with `LLVM ERROR: Instruction Combining did not reach a fixpoint after 1 iterations` — known pocl 6.0 limitation hit by complex kernels using OpenCL.std math intrinsics. Works on pocl 7.2-pre and Intel GPU.
- **Verified working on**: pocl 6.0 (CPU), pocl 7.2-pre (CPU; `nbody` on aarch64 needs the alignment patch from pocl#2174 above to match host layout for `cl::Double3`), Intel Gen11 GPU.
