#![cfg_attr(target_arch = "spirv", no_std)]

use glam::USizeVec3;
use num_complex::Complex32;
use spirv_std::{glam, spirv};

/// Viewport parameters for the Mandelbrot computation.
#[derive(Copy, Clone)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
    pub cx_min: f32,
    pub cx_max: f32,
    pub cy_min: f32,
    pub cy_max: f32,
}

/// Compute the Mandelbrot iteration count for a point `c` in the
/// complex plane. Returns the number of iterations before escape, or
/// `max_iter` if the point is in the set.
fn mandelbrot(c: Complex32, max_iter: u32) -> u32 {
    let mut z = Complex32::new(0.0, 0.0);
    let mut i = 0u32;
    while i < max_iter {
        if z.norm_sqr() > 4.0 {
            return i;
        }
        z = z * z + c;
        i += 1;
    }
    max_iter
}

/// Compute the Mandelbrot set for a 2D grid of pixels.
///
/// Each work item computes one pixel. The output buffer is a flat array
/// of iteration counts in row-major order.
#[spirv(kernel)]
pub fn mandelbrot_kernel(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
    vp: Viewport,
    max_iter: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;

    if px >= vp.width || py >= vp.height {
        return;
    }

    let c = Complex32::new(
        vp.cx_min + (px as f32 / vp.width as f32) * (vp.cx_max - vp.cx_min),
        vp.cy_min + (py as f32 / vp.height as f32) * (vp.cy_max - vp.cy_min),
    );

    let index = (py * vp.width + px) as usize;
    output[index] = mandelbrot(c, max_iter);
}
