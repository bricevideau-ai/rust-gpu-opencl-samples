use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{
    Buffer, CL_MEM_OBJECT_IMAGE2D, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_RGBA, CL_UNSIGNED_INT8,
    ClMem, Image,
};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id, cl_image_desc, cl_image_format};
use spirv_builder::{Capability, CompileResult, ShaderPanicStrategy, SpirvBuilder};
use std::path::Path;
use std::ptr;
use std::time::{Duration, Instant};

// ── OpenCL helpers ─────────────────────────────────────────────────────

struct OclContext {
    device_id: cl_device_id,
    context: Context,
    queue: CommandQueue,
}

impl OclContext {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_ALL)?
            .first()
            .expect("no OpenCL devices found");
        let device = Device::new(device_id);
        println!("Device:  {} ({})", device.name()?, device.vendor()?);
        println!("Version: {}", device.version()?);
        let context = Context::from_device(&device)?;
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;
        Ok(Self {
            device_id,
            context,
            queue,
        })
    }

    fn build_program(&self, spv_bytes: &[u8]) -> Result<Program, Box<dyn std::error::Error>> {
        let mut program = Program::create_from_il(&self.context, spv_bytes)
            .map_err(|e| format!("create_from_il: {e}"))?;
        if let Err(e) = program.build(self.context.devices(), "") {
            let log = program
                .get_build_log(self.device_id)
                .unwrap_or_else(|_| "no build log".into());
            return Err(format!("program.build: {e}\nbuild log: {log}").into());
        }
        Ok(program)
    }

    fn upload<T>(&self, data: &[T]) -> Result<DeviceSlice<T>, Box<dyn std::error::Error>> {
        let mut buffer = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                data.len(),
                ptr::null_mut(),
            )?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])?
                .wait()?;
        }
        Ok(DeviceSlice {
            buffer,
            len: data.len(),
        })
    }

    /// Allocate a device buffer for `len` `T`s with no host-side
    /// initialisation. Wraps `Buffer::create(null_mut())` so callers
    /// don't need an `unsafe` block — `null_mut()` makes OpenCL
    /// allocate fresh device memory and ignore the host_ptr entirely,
    /// avoiding the dereference contract that makes `Buffer::create`
    /// generally unsafe.
    fn alloc<T>(&self, len: usize) -> Result<DeviceSlice<T>, Box<dyn std::error::Error>> {
        let buffer =
            unsafe { Buffer::<T>::create(&self.context, CL_MEM_READ_WRITE, len, ptr::null_mut())? };
        Ok(DeviceSlice { buffer, len })
    }

    fn download<T>(
        &self,
        src: &DeviceSlice<T>,
        dst: &mut [T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            self.queue
                .enqueue_read_buffer(&src.buffer, CL_BLOCKING, 0, dst, &[])?
                .wait()?;
        }
        Ok(())
    }

    fn run(
        &self,
        kernel: &Kernel,
        global_work_size: &[usize],
        local_work_size: Option<&[usize]>,
        args: &[&dyn KernelArg],
    ) -> Result<Event, Box<dyn std::error::Error>> {
        let mut exec = ExecuteKernel::new(kernel);
        for arg in args {
            arg.set(&mut exec);
        }
        match global_work_size.len() {
            1 => {
                exec.set_global_work_size(global_work_size[0]);
            }
            2 => {
                exec.set_global_work_sizes(&[global_work_size[0], global_work_size[1]]);
            }
            3 => {
                exec.set_global_work_sizes(&[
                    global_work_size[0],
                    global_work_size[1],
                    global_work_size[2],
                ]);
            }
            _ => return Err("invalid work dimensions".into()),
        }
        if let Some(lws) = local_work_size {
            match lws.len() {
                1 => {
                    exec.set_local_work_size(lws[0]);
                }
                2 => {
                    exec.set_local_work_sizes(&[lws[0], lws[1]]);
                }
                3 => {
                    exec.set_local_work_sizes(&[lws[0], lws[1], lws[2]]);
                }
                _ => return Err("invalid local work dimensions".into()),
            }
        }
        let event = unsafe { exec.enqueue_nd_range(&self.queue)? };
        event.wait()?;
        Ok(event)
    }
}

struct DeviceSlice<T> {
    buffer: Buffer<T>,
    len: usize,
}

impl<T> DeviceSlice<T> {
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.len
    }
}

trait KernelArg {
    fn set(&self, exec: &mut ExecuteKernel<'_>);
}

impl<T> KernelArg for DeviceSlice<T> {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        let len: usize = self.len;
        unsafe {
            exec.set_arg(&self.buffer).set_arg(&len);
        }
    }
}

impl KernelArg for u32 {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg(self);
        }
    }
}

impl KernelArg for f32 {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg(self);
        }
    }
}

impl KernelArg for f64 {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg(self);
        }
    }
}

/// Viewport struct matching the kernel's `Viewport` type layout.
#[repr(C)]
#[derive(Copy, Clone)]
struct Viewport {
    width: u32,
    height: u32,
    cx_min: f32,
    cx_max: f32,
    cy_min: f32,
    cy_max: f32,
}

impl KernelArg for Viewport {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg(self);
        }
    }
}

/// A local (workgroup) memory allocation, set as a kernel argument via `clSetKernelArg(NULL)`.
/// Used when workgroup memory is a kernel parameter rather than a module-scope variable.
#[allow(dead_code)]
struct LocalBuffer {
    size_bytes: usize,
}

impl KernelArg for LocalBuffer {
    fn set(&self, exec: &mut ExecuteKernel<'_>) {
        unsafe {
            exec.set_arg_local_buffer(self.size_bytes);
        }
    }
}

fn compile_kernel(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2")
        .shader_panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit {
            print_inputs: true,
            print_backtrace: true,
        })
        .build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

/// OpenCL 1.2 + `Float64` capability (opt-in for the OpenCL SPIR-V env;
/// not all OpenCL targets require it). Uses the same DebugPrintfThenExit
/// abort strategy as the default `compile_kernel` so panic messages
/// surface on the host.
fn compile_kernel_f64(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2")
        .capability(Capability::Float64)
        .shader_panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit {
            print_inputs: true,
            print_backtrace: true,
        })
        .build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

fn compile_kernel_opencl2(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    // Barrier-using kernels need unreachable: the debug-printf strategy's OpReturn
    // can cause work item divergence at barriers, which is UB (PoCL #2156).
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl2.0")
        .capability(Capability::Groups)
        .shader_panic_strategy(
            ShaderPanicStrategy::UNSOUND_DO_NOT_USE_UndefinedBehaviorViaUnreachable,
        )
        .build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

fn profiling_duration(event: &Event) -> Option<Duration> {
    let start = event.profiling_command_start().ok()?;
    let end = event.profiling_command_end().ok()?;
    Some(Duration::from_nanos(end - start))
}

// ── Samples ────────────────────────────────────────────────────────────

fn run_collatz(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ Collatz ═══");
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/collatz");
    let (spv_bytes, compile_time) = compile_kernel(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "collatz_kernel")?;

    let top = 2u32.pow(20);
    let src_range = 1..top;
    let mut data: Vec<u32> = src_range.clone().collect();
    let n = data.len();

    let buf = ocl.upload(&data)?;
    let event = ocl.run(&kernel, &[n], None, &[&buf])?;
    ocl.download(&buf, &mut data)?;

    if let Some(d) = profiling_duration(&event) {
        println!("Kernel:  {d:?}");
    }

    // Validate.
    let checks: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (27, 111)];
    let mut ok = true;
    for &(input, expected) in checks {
        let got = data[(input - 1) as usize];
        if got != expected {
            eprintln!("FAIL: collatz({input}) = {got}, expected {expected}");
            ok = false;
        }
    }
    if ok {
        println!("Verify:  passed");
    }

    // Print record-holders.
    println!("1: 0");
    let mut max = 0;
    for (src, out) in src_range.zip(data.iter().copied()) {
        if out == u32::MAX {
            println!("{src}: overflowed");
            break;
        } else if out > max {
            max = out;
            println!("{src}: {out}");
        }
    }
    Ok(())
}

fn run_mandelbrot(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ Mandelbrot ═══");
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/mandelbrot");
    let (spv_bytes, compile_time) = compile_kernel(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "mandelbrot_kernel")?;

    let width: u32 = 80;
    let height: u32 = 40;
    let max_iter: u32 = 100;

    let vp = Viewport {
        width,
        height,
        cx_min: -2.0,
        cx_max: 1.0,
        cy_min: -1.0,
        cy_max: 1.0,
    };

    let n = (width * height) as usize;
    let mut pixels = vec![0u32; n];

    let buf = ocl.upload(&pixels)?;
    let event = ocl.run(
        &kernel,
        &[width as usize, height as usize],
        None,
        &[&buf, &vp, &max_iter],
    )?;
    ocl.download(&buf, &mut pixels)?;

    if let Some(d) = profiling_duration(&event) {
        println!("Kernel:  {d:?} ({width}x{height}, max_iter={max_iter})");
    }

    // Render as ASCII art.
    let palette = b" .:-=+*#%@";
    for row in 0..height {
        let mut line = String::with_capacity(width as usize);
        for col in 0..width {
            let iter = pixels[(row * width + col) as usize];
            let ch = if iter >= max_iter {
                ' '
            } else {
                palette[(iter as usize) % palette.len()] as char
            };
            line.push(ch);
        }
        println!("{line}");
    }

    Ok(())
}

fn run_reduce(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ Hierarchical Reduction ═══");
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/reduce");
    let (spv_bytes, compile_time) = compile_kernel_opencl2(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "reduce_kernel")?;

    const WG_SIZE: usize = 32;

    let num_workgroups = 4;
    let n = WG_SIZE * num_workgroups; // 128 elements
    let input: Vec<u32> = (1..=n as u32).collect();

    let input_buf = ocl.upload(&input)?;
    let output_buf = ocl.upload(&vec![0u32; num_workgroups])?;

    let event = ocl.run(&kernel, &[n], Some(&[WG_SIZE]), &[&input_buf, &output_buf])?;

    let mut output = vec![0u32; num_workgroups];
    ocl.download(&output_buf, &mut output)?;

    if let Some(d) = profiling_duration(&event) {
        println!("Kernel:  {d:?} ({n} elements, {num_workgroups} workgroups of {WG_SIZE})");
    }

    // Compute CPU reference (sum per workgroup).
    let mut expected = vec![0u32; num_workgroups];
    for (wg, total) in expected.iter_mut().enumerate() {
        let base = wg * WG_SIZE;
        *total = input[base..base + WG_SIZE].iter().sum();
    }

    // Verify.
    let mut ok = true;
    for (wg, (got, exp)) in output.iter().zip(expected.iter()).enumerate() {
        if got != exp {
            eprintln!("FAIL wg[{wg}]: got {got}, expected {exp}");
            ok = false;
        }
    }
    if ok {
        println!("Verify:  passed ({num_workgroups} workgroup reductions)");
    }

    // Print results.
    println!("Workgroup sums: {output:?}");
    println!("Expected:       {expected:?}");

    Ok(())
}

fn run_debug_abort(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ Debug Abort (DebugPrintf strategy) ═══");
    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/collatz");
    let (spv_bytes, compile_time) = compile_kernel(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());
    println!("Strategy: debug-printf (abort blocks → OpenCL printf + OpReturn)");

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "collatz_kernel")?;

    // Allocate a buffer smaller than global_size. Out-of-bounds work items
    // hit a bounds check → printf diagnostic + OpReturn (early exit).
    let global_size = 128;
    let buf_len = 64;
    let mut data: Vec<u32> = (1..=buf_len as u32).collect();

    let buf = ocl.upload(&data)?;
    println!("Launching {global_size} work items with buffer length {buf_len}...");
    let event = ocl.run(&kernel, &[global_size], None, &[&buf])?;
    ocl.download(&buf, &mut data)?;

    if let Some(d) = profiling_duration(&event) {
        println!("Kernel:  {d:?}");
    }

    // Verify the in-bounds portion still computed correctly.
    let checks: &[(u32, u32)] = &[(1, 0), (2, 1), (3, 7), (27, 111)];
    let mut ok = true;
    for &(input, expected) in checks {
        let got = data[(input - 1) as usize];
        if got != expected {
            eprintln!("FAIL: collatz({input}) = {got}, expected {expected}");
            ok = false;
        }
    }
    if ok {
        println!("Verify:  in-bounds elements computed correctly");
    }

    Ok(())
}

fn compile_kernel_image(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2").build()?;
    let spv_path = result.module.unwrap_single();
    let spv_bytes = std::fs::read(spv_path)?;
    Ok((spv_bytes, start.elapsed()))
}

fn run_mandelbrot_image(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ Mandelbrot Image ═══");

    let device = Device::new(ocl.device_id);
    let has_images = device.image_support().unwrap_or(false);
    if !has_images {
        println!("Skipped: device does not support images");
        return Ok(());
    }

    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/mandelbrot-image");
    let (spv_bytes, compile_time) = compile_kernel_image(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "mandelbrot_image")?;

    let width: u32 = 1920;
    let height: u32 = 1080;
    let max_iter: u32 = 256;

    let format = cl_image_format {
        image_channel_order: CL_RGBA,
        image_channel_data_type: CL_UNSIGNED_INT8,
    };
    let desc = cl_image_desc {
        image_type: CL_MEM_OBJECT_IMAGE2D,
        image_width: width as usize,
        image_height: height as usize,
        image_depth: 0,
        image_array_size: 0,
        image_row_pitch: 0,
        image_slice_pitch: 0,
        num_mip_levels: 0,
        num_samples: 0,
        buffer: ptr::null_mut(),
    };

    let image = unsafe {
        Image::create(
            &ocl.context,
            CL_MEM_WRITE_ONLY,
            &format,
            &desc,
            ptr::null_mut(),
        )?
    };

    let mut exec = ExecuteKernel::new(&kernel);
    let cl_mem_handle = image.get();
    unsafe {
        exec.set_arg(&cl_mem_handle)
            .set_arg(&width)
            .set_arg(&height)
            .set_arg(&max_iter);
    }
    let event = unsafe {
        exec.set_global_work_sizes(&[width as usize, height as usize])
            .enqueue_nd_range(&ocl.queue)?
    };
    event.wait()?;

    if let Some(d) = profiling_duration(&event) {
        println!("Kernel:  {d:?} ({width}x{height}, {max_iter} iterations)");
    }

    let pixel_count = (width * height) as usize;
    let mut pixels = vec![0u8; pixel_count * 4];
    let origin = [0usize, 0, 0];
    let region = [width as usize, height as usize, 1];
    unsafe {
        ocl.queue
            .enqueue_read_image(
                &image,
                CL_BLOCKING,
                origin.as_ptr(),
                region.as_ptr(),
                0,
                0,
                pixels.as_mut_ptr().cast(),
                &[],
            )?
            .wait()?;
    }

    let ppm_path = "mandelbrot.ppm";
    let mut ppm = Vec::with_capacity(pixel_count * 3 + 64);
    ppm.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for chunk in pixels.chunks_exact(4) {
        ppm.push(chunk[0]);
        ppm.push(chunk[1]);
        ppm.push(chunk[2]);
    }
    std::fs::write(ppm_path, &ppm)?;
    println!(
        "Output:  {ppm_path} ({width}x{height}, {} bytes)",
        ppm.len()
    );

    Ok(())
}

// Reuse the kernel crate's types and host-side helpers — the verifier
// below reads identically to the kernel because it calls the same
// `dot`/`length` symbols. `Body` is `#[repr(C)] { Double3, Double3, f64 }`
// → 96 bytes / 32-byte aligned (asserted in the kernel crate).
use nbody::{Body, dot, length};
use spirv_std::cl::Double3;

/// Total kinetic energy of the system. Used as a sanity check on the
/// integrator: a leapfrog step with reasonable dt should preserve total
/// energy to within a small drift over thousands of steps.
fn kinetic(bodies: &[Body]) -> f64 {
    bodies
        .iter()
        .map(|b| 0.5 * b.mass * dot(b.vel, b.vel))
        .sum()
}

/// Total gravitational potential energy of the system.
///   U = -G · Σᵢ<ⱼ mᵢ·mⱼ / |xᵢ - xⱼ|
/// Includes the same Plummer softening as the kernel so KE+PE values
/// are comparable to the kernel's energy.
fn potential(bodies: &[Body]) -> f64 {
    const G: f64 = 1.0;
    const SOFTENING_SQ: f64 = 0.01;
    let mut u = 0.0;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let d = bodies[j].pos - bodies[i].pos;
            let r2 = dot(d, d) + SOFTENING_SQ;
            u -= G * bodies[i].mass * bodies[j].mass / r2.sqrt();
        }
    }
    u
}

fn total_momentum(bodies: &[Body]) -> Double3 {
    let mut p = Double3::default();
    for b in bodies {
        p += b.vel * b.mass;
    }
    p
}

fn run_nbody(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ N-body (f64 vectors) ═══");

    // The kernel uses double-precision throughout; bail early on
    // devices that don't advertise `cl_khr_fp64`. Many integrated GPUs
    // (e.g. Intel Iris) report OpenCL 3.0 but no fp64 support.
    let device = Device::new(ocl.device_id);
    let extensions = device.extensions()?;
    if !extensions.contains("cl_khr_fp64") {
        println!("SKIP: device does not support cl_khr_fp64");
        return Ok(());
    }

    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/nbody");
    let (spv_bytes, compile_time) = compile_kernel_f64(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "step")?;

    // Initial conditions: 3-body figure-8 orbit (Chenciner & Montgomery
    // 2000), the canonical zero-momentum bound 3-body solution. With
    // unit masses, a body at position (0.97000436, -0.24308753, 0)
    // moving with velocity (0.466203685, 0.43236573, 0), and the
    // mirror-symmetric configuration on the other side. We follow the
    // standard normalisation: third body at the origin moves with
    // velocity (-0.93240737, -0.86473146, 0).
    let initial = vec![
        Body::new(
            [0.97000436, -0.24308753, 0.0],
            [0.466_203_685, 0.432_365_73, 0.0],
            1.0,
        ),
        Body::new(
            [-0.97000436, 0.24308753, 0.0],
            [0.466_203_685, 0.432_365_73, 0.0],
            1.0,
        ),
        Body::new([0.0, 0.0, 0.0], [-0.932_407_37, -0.864_731_46, 0.0], 1.0),
        // Add a few extra bodies so the kernel has more than the
        // canonical figure-8 to chew on. These live far enough out
        // that they don't disrupt the central 3-body dance much over
        // the simulation window.
        Body::at_rest([5.0, 0.0, 0.0], 0.1),
        Body::at_rest([-5.0, 0.0, 0.0], 0.1),
        Body::at_rest([0.0, 5.0, 0.0], 0.1),
        Body::at_rest([0.0, -5.0, 0.0], 0.1),
    ];
    let n = initial.len();
    println!("Bodies:  {n}");

    let buf_a = ocl.upload(&initial)?;
    let buf_b: DeviceSlice<Body> = ocl.alloc(n)?;

    let dt: f64 = 0.001;
    let steps: usize = 1_000;

    let energy_initial = kinetic(&initial) + potential(&initial);
    let momentum_initial = total_momentum(&initial);
    println!(
        "Initial: E = {energy_initial:+.6}, |p| = {:+.6e}",
        length(momentum_initial)
    );

    // Ping-pong between the two buffers. Even step: read A, write B;
    // odd step: read B, write A. After `steps` iterations the latest
    // state lives in `final_buf`.
    let (mut from_buf, mut to_buf) = (&buf_a, &buf_b);
    let mut total_kernel_time = Duration::ZERO;
    for _ in 0..steps {
        let event = ocl.run(&kernel, &[n], None, &[from_buf, to_buf, &dt])?;
        if let Some(d) = profiling_duration(&event) {
            total_kernel_time += d;
        }
        std::mem::swap(&mut from_buf, &mut to_buf);
    }
    let final_buf = from_buf;

    let mut final_state = vec![Body::at_rest([0.0; 3], 0.0); n];
    ocl.download(final_buf, &mut final_state)?;

    println!("Total kernel time: {total_kernel_time:?} ({steps} steps)");

    let energy_final = kinetic(&final_state) + potential(&final_state);
    let momentum_final = total_momentum(&final_state);
    println!(
        "Final:   E = {energy_final:+.6}, |p| = {:+.6e}",
        length(momentum_final)
    );

    let energy_drift = (energy_final - energy_initial) / energy_initial.abs();
    println!("Energy drift: {:+.3e}", energy_drift);

    // Validation: the leapfrog integrator with these dt + softening
    // should keep relative energy drift well under 5% over the run,
    // and total momentum drift small (the asymmetric extra bodies
    // accumulate `f64` round-off but stay within ~1e-3).
    let mut ok = true;
    if energy_drift.abs() > 0.05 {
        eprintln!("FAIL: energy drift {energy_drift:.3e} > 5%");
        ok = false;
    }
    let p_drift = length(momentum_final - momentum_initial);
    if p_drift > 1e-2 {
        eprintln!("FAIL: momentum drift {p_drift:.3e} > 1e-2");
        ok = false;
    }

    println!("{}", if ok { "OK" } else { "FAIL" });
    Ok(())
}

fn run_raymarch(ocl: &OclContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n═══ Ray-marched SDF ═══");

    let device = Device::new(ocl.device_id);
    let has_images = device.image_support().unwrap_or(false);
    if !has_images {
        println!("Skipped: device does not support images");
        return Ok(());
    }

    let kernel_crate = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels/raymarch");
    let (spv_bytes, compile_time) = compile_kernel_image(&kernel_crate)?;
    println!("Compiled: {} bytes, {compile_time:?}", spv_bytes.len());

    let program = ocl.build_program(&spv_bytes)?;
    let kernel = Kernel::create(&program, "raymarch")?;

    let width: u32 = 1280;
    let height: u32 = 720;

    let format = cl_image_format {
        image_channel_order: CL_RGBA,
        image_channel_data_type: CL_UNSIGNED_INT8,
    };
    let desc = cl_image_desc {
        image_type: CL_MEM_OBJECT_IMAGE2D,
        image_width: width as usize,
        image_height: height as usize,
        image_depth: 0,
        image_array_size: 0,
        image_row_pitch: 0,
        image_slice_pitch: 0,
        num_mip_levels: 0,
        num_samples: 0,
        buffer: ptr::null_mut(),
    };

    let image = unsafe {
        Image::create(
            &ocl.context,
            CL_MEM_WRITE_ONLY,
            &format,
            &desc,
            ptr::null_mut(),
        )?
    };

    let mut exec = ExecuteKernel::new(&kernel);
    let cl_mem_handle = image.get();
    unsafe {
        exec.set_arg(&cl_mem_handle)
            .set_arg(&width)
            .set_arg(&height);
    }
    let event = unsafe {
        exec.set_global_work_sizes(&[width as usize, height as usize])
            .enqueue_nd_range(&ocl.queue)?
    };
    event.wait()?;

    if let Some(d) = profiling_duration(&event) {
        println!("Kernel:  {d:?} ({width}x{height})");
    }

    let pixel_count = (width * height) as usize;
    let mut pixels = vec![0u8; pixel_count * 4];
    let origin = [0usize, 0, 0];
    let region = [width as usize, height as usize, 1];
    unsafe {
        ocl.queue
            .enqueue_read_image(
                &image,
                CL_BLOCKING,
                origin.as_ptr(),
                region.as_ptr(),
                0,
                0,
                pixels.as_mut_ptr().cast(),
                &[],
            )?
            .wait()?;
    }

    let ppm_path = "raymarch.ppm";
    let mut ppm = Vec::with_capacity(pixel_count * 3 + 64);
    ppm.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    for chunk in pixels.chunks_exact(4) {
        ppm.push(chunk[0]);
        ppm.push(chunk[1]);
        ppm.push(chunk[2]);
    }
    std::fs::write(ppm_path, &ppm)?;
    println!(
        "Output:  {ppm_path} ({width}x{height}, {} bytes)",
        ppm.len()
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocl = OclContext::new()?;

    let sample = std::env::args().nth(1);
    match sample.as_deref() {
        Some("collatz") => run_collatz(&ocl)?,
        Some("mandelbrot") => run_mandelbrot(&ocl)?,
        Some("reduce") => run_reduce(&ocl)?,
        Some("debug-abort") => run_debug_abort(&ocl)?,
        Some("mandelbrot-image") => run_mandelbrot_image(&ocl)?,
        Some("raymarch") => run_raymarch(&ocl)?,
        Some("nbody") => run_nbody(&ocl)?,
        _ => {
            run_collatz(&ocl)?;
            run_mandelbrot(&ocl)?;
            run_reduce(&ocl)?;
            run_mandelbrot_image(&ocl)?;
            run_raymarch(&ocl)?;
            run_nbody(&ocl)?;
        }
    }

    Ok(())
}
