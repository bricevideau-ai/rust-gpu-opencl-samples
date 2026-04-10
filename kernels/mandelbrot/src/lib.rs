#![cfg_attr(target_arch = "spirv", no_std)]

use glam::USizeVec3;
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

/// Compute the Mandelbrot iteration count for a point (cx, cy) in the
/// complex plane. Returns the number of iterations before escape, or
/// `max_iter` if the point is in the set.
fn mandelbrot(cx: f32, cy: f32, max_iter: u32) -> u32 {
    let mut zx = 0.0f32;
    let mut zy = 0.0f32;
    let mut i = 0u32;
    while i < max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > 4.0 {
            return i;
        }
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
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

    let cx = vp.cx_min + (px as f32 / vp.width as f32) * (vp.cx_max - vp.cx_min);
    let cy = vp.cy_min + (py as f32 / vp.height as f32) * (vp.cy_max - vp.cy_min);

    let index = (py * vp.width + px) as usize;
    output[index] = mandelbrot(cx, cy, max_iter);
}
