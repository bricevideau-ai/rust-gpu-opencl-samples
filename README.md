# rust-gpu OpenCL Samples

OpenCL compute kernels written in Rust using [rust-gpu](https://github.com/Rust-GPU/rust-gpu).

## Samples

### Collatz
Computes the length of the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture) for each integer 1..2²⁰, finding record-holders ([OEIS A006877](https://oeis.org/A006877)).

### Mandelbrot
Computes the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) and renders it as ASCII art. Uses `printf!` to print viewport parameters from the GPU.

### Reduce
Hierarchical reduction using subgroup operations (`group_i_add`) and shared/local memory. Demonstrates subgroup builtins, workgroup barriers, and cross-subgroup communication.

## Requirements

- Rust nightly (see `rust-toolchain.toml`)
- An OpenCL runtime with SPIR-V IL support (e.g., pocl 6.0+, Intel OpenCL GPU/CPU runtime)

## Running

```bash
# Run all samples
cargo run -p runner --release

# Run a specific sample
cargo run -p runner --release -- collatz
cargo run -p runner --release -- mandelbrot
cargo run -p runner --release -- reduce
cargo run -p runner --release -- debug-abort  # debug-printf abort strategy demo
```

## Project Structure

```
kernels/
  collatz/     — Collatz conjecture kernel (#[spirv(kernel)])
  mandelbrot/  — Mandelbrot set kernel with printf (#[spirv(kernel)])
  reduce/      — Hierarchical reduction with subgroup ops (OpenCL 2.0)
runner/        — Host-side OpenCL runner (compiles + executes kernels)
```

Each kernel crate is a `dylib` compiled to OpenCL SPIR-V via `spirv-builder`. The runner uses the `opencl3` crate to load and execute the kernels.
