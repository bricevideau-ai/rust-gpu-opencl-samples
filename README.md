# rust-gpu OpenCL Samples

OpenCL compute kernels written in Rust using [rust-gpu](https://github.com/Rust-GPU/rust-gpu).

## Samples

### Collatz
Computes the length of the [Collatz sequence](https://en.wikipedia.org/wiki/Collatz_conjecture) for each integer 1..2²⁰, finding record-holders ([OEIS A006877](https://oeis.org/A006877)).

### Mandelbrot
Computes the [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) and renders it as ASCII art.

## Requirements

- Rust nightly (see `rust-toolchain.toml`)
- `spirv-tools` installed (`spirv-val`, `spirv-opt`)
- An OpenCL runtime with SPIR-V IL support (e.g., pocl 6.0+, Intel OpenCL CPU runtime)

## Running

```bash
# Run all samples
cargo run -p runner --release

# Run a specific sample
cargo run -p runner --release -- collatz
cargo run -p runner --release -- mandelbrot
```

## Project Structure

```
kernels/
  collatz/     — Collatz conjecture kernel (#[spirv(kernel)])
  mandelbrot/  — Mandelbrot set kernel (#[spirv(kernel)])
runner/        — Host-side OpenCL runner (compiles + executes kernels)
```

Each kernel crate is a `dylib` compiled to OpenCL SPIR-V via `spirv-builder`. The runner uses the `opencl3` crate to load and execute the kernels.
