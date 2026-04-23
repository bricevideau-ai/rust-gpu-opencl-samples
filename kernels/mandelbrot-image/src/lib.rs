#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{USizeVec3, UVec4, Vec4};
use spirv_std::{Image, glam, spirv};

fn mandelbrot_color(ix: u32, max_iter: u32) -> UVec4 {
    if ix >= max_iter {
        return UVec4::ZERO;
    }
    let t = ix as f32 / max_iter as f32;
    let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u32;
    let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u32;
    let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u32;
    UVec4::new(r, g, b, 255)
}

#[spirv(kernel)]
pub fn mandelbrot_image(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &mut Image!(2D, type=u32, sampled=false),
    width: u32,
    height: u32,
    max_iter: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;
    if px >= width || py >= height {
        return;
    }

    let cx = -2.5 + (px as f32 / width as f32) * 3.5;
    let cy = -1.25 + (py as f32 / height as f32) * 2.5;

    let mut zx = 0.0f32;
    let mut zy = 0.0f32;
    let mut ix = 0u32;
    while ix < max_iter && zx * zx + zy * zy < 4.0 {
        let tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        ix += 1;
    }

    let color = mandelbrot_color(ix, max_iter);
    let coord = glam::IVec2::new(px as i32, py as i32);
    unsafe {
        image.write(coord, color);
    }
}

#[spirv(kernel)]
pub fn fill_gradient(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &mut Image!(2D, type=f32, sampled=false),
    width: u32,
    height: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;
    if px >= width || py >= height {
        return;
    }

    let r = px as f32 / width as f32;
    let g = py as f32 / height as f32;
    let color = Vec4::new(r, g, 0.2, 1.0);
    let coord = glam::IVec2::new(px as i32, py as i32);
    unsafe {
        image.write(coord, color);
    }
}
