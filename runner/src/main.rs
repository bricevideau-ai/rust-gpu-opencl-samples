use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ALL, Device, get_all_devices};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_device_id};
use spirv_builder::{CompileResult, SpirvBuilder};
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
        args: &[&dyn KernelArg],
    ) -> Result<Event, Box<dyn std::error::Error>> {
        let mut exec = ExecuteKernel::new(kernel);
        for arg in args {
            arg.set(&mut exec);
        }
        let event = unsafe {
            match global_work_size.len() {
                1 => exec
                    .set_global_work_size(global_work_size[0])
                    .enqueue_nd_range(&self.queue)?,
                2 => exec
                    .set_global_work_sizes(&[global_work_size[0], global_work_size[1]])
                    .enqueue_nd_range(&self.queue)?,
                3 => exec
                    .set_global_work_sizes(&[
                        global_work_size[0],
                        global_work_size[1],
                        global_work_size[2],
                    ])
                    .enqueue_nd_range(&self.queue)?,
                _ => return Err("invalid work dimensions".into()),
            }
        };
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

fn compile_kernel(path: &Path) -> Result<(Vec<u8>, Duration), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let result: CompileResult = SpirvBuilder::new(path, "spirv-unknown-opencl1.2").build()?;
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
    let event = ocl.run(&kernel, &[n], &[&buf])?;
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
    let cx_min: f32 = -2.0;
    let cx_max: f32 = 1.0;
    let cy_min: f32 = -1.0;
    let cy_max: f32 = 1.0;

    let n = (width * height) as usize;
    let mut pixels = vec![0u32; n];

    let buf = ocl.upload(&pixels)?;
    let event = ocl.run(
        &kernel,
        &[width as usize, height as usize],
        &[
            &buf, &width, &height, &max_iter, &cx_min, &cx_max, &cy_min, &cy_max,
        ],
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocl = OclContext::new()?;

    let sample = std::env::args().nth(1);
    match sample.as_deref() {
        Some("collatz") => run_collatz(&ocl)?,
        Some("mandelbrot") => run_mandelbrot(&ocl)?,
        _ => {
            run_collatz(&ocl)?;
            run_mandelbrot(&ocl)?;
        }
    }

    Ok(())
}
