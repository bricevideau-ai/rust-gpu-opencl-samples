# rust-gpu OpenCL Samples

OpenCL compute kernels written in Rust using [rust-gpu](https://github.com/Rust-GPU/rust-gpu) on the [`opencl-kernel-support`](https://github.com/bricevideau-ai/rust-gpu/tree/opencl-kernel-support) branch.

## Samples

### Collatz
Computes the length of the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture) for each integer 1..2²⁰, finding record-holders ([OEIS A006877](https://oeis.org/A006877)).

### Mandelbrot
Computes the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) and renders it as ASCII art. Uses `printf!` to print viewport parameters from the GPU.

### Mandelbrot-image
Same Mandelbrot computation rendered to a Full-HD OpenCL 2D image (`Image!(2D, type=u32, sampled=false)`) and saved as a PPM. Exercises image kernel arguments and `image.write()`.

### Reduce
Hierarchical reduction using subgroup operations (`group_i_add`) and shared/local memory. Demonstrates subgroup builtins, workgroup barriers, and cross-subgroup communication. Requires OpenCL 2.0 + `Groups` capability.

### Raymarch
SDF ray marcher (two soft-min'd spheres above a ground plane, sun lighting + soft shadows + distance fog) written end-to-end in `spirv_std::cl::Float3` so every per-pixel operation lowers to native `OpTypeVector` / `OpDot` / `OpExtInst` codegen — no per-component scalarisation through glam's scalar fallback. Exercises the OpenCL.std math intrinsics (`sqrt`/`cos`/`sin`/`pow`/`exp`/`length`/`normalize`/etc.) heavily, and the host crate ships a one-pixel host-vs-GPU bit-for-bit smoke test.

### N-body
Direct (O(N²)) gravitational n-body simulation using `spirv_std::cl::Double3` (the OpenCL `double3` ABI: 32-byte aligned, 32-byte size). Each kernel invocation advances the simulation by one timestep using a leapfrog (kick-drift-kick) integrator. Exercises arrays of structs containing native `cl::*` vectors as a `cross_workgroup` parameter.

### Debug-abort
Demo of the `debug-printf` abort strategy on Kernel targets — the post-link pass converts the SPIR-T-emitted `NonSemantic.DebugPrintf` into `OpenCL.std` printf so abort diagnostics are visible on stock OpenCL runtimes (no `NonSemantic` extension required). Not in the default run set.

## Requirements

- Rust nightly (see `rust-toolchain.toml`)
- An OpenCL runtime with SPIR-V IL support (e.g. pocl 6.0+, Intel OpenCL GPU/CPU runtime)
- Image samples (`mandelbrot-image`) need a device that reports `image_support`
- `reduce` needs an OpenCL 2.0 device that supports the `Groups` capability

## Running

```bash
# Run the default set (every sample except debug-abort)
cargo run -p runner --release

# Run a specific sample
cargo run -p runner --release -- collatz
cargo run -p runner --release -- mandelbrot
cargo run -p runner --release -- mandelbrot-image
cargo run -p runner --release -- reduce
cargo run -p runner --release -- raymarch
cargo run -p runner --release -- nbody
cargo run -p runner --release -- debug-abort   # opt-in only
```

## Project structure

```
kernels/
  collatz/           — Collatz conjecture kernel (#[spirv(kernel)])
  mandelbrot/        — Mandelbrot set kernel with printf
  mandelbrot-image/  — Mandelbrot rendered to an OpenCL 2D image (PPM out)
  reduce/            — Hierarchical reduction with subgroup ops (OpenCL 2.0)
  raymarch/          — SDF ray marcher built on cl::Float3 + opencl_std::*
  nbody/             — N-body simulation built on cl::Double3
runner/              — Host-side OpenCL runner (compiles + executes kernels)
```

Each kernel crate is a `dylib` compiled to OpenCL SPIR-V via `spirv-builder`. The runner uses the `opencl3` crate to load and execute the kernels.
