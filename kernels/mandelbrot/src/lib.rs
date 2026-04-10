#![cfg_attr(target_arch = "spirv", no_std)]

use glam::USizeVec3;
use spirv_std::{glam, spirv};

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
///
/// Parameters:
/// - `id`: global invocation ID (x = column, y = row)
/// - `output`: iteration count per pixel (width * height elements)
/// - `width`: image width in pixels
/// - `height`: image height in pixels
/// - `max_iter`: maximum iteration count
/// - `cx_min`, `cx_max`, `cy_min`, `cy_max`: complex plane bounds
#[spirv(kernel)]
pub fn mandelbrot_kernel(
    #[spirv(global_invocation_id)] id: USizeVec3,
    #[spirv(cross_workgroup)] output: &mut [u32],
    width: u32,
    height: u32,
    max_iter: u32,
    cx_min: f32,
    cx_max: f32,
    cy_min: f32,
    cy_max: f32,
) {
    let px = id.x as u32;
    let py = id.y as u32;

    if px >= width || py >= height {
        return;
    }

    let cx = cx_min + (px as f32 / width as f32) * (cx_max - cx_min);
    let cy = cy_min + (py as f32 / height as f32) * (cy_max - cy_min);

    let index = (py * width + px) as usize;
    output[index] = mandelbrot(cx, cy, max_iter);
}
