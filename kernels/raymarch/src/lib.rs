//! SDF ray marcher rendering two soft-min'd spheres above a ground plane,
//! with sun lighting + soft shadows + distance fog. Written entirely in
//! `spirv_std::cl::Float3` so every per-pixel operation is guaranteed to
//! lower to the native `OpTypeVector` / `OpDot` / `OpExtInst` codegen path
//! — no per-component scalarisation through `glam`'s scalar fallback.
//!
//! Same code runs on host (via the `Componentwise` host arms in
//! `opencl_std::*`) so the `runner` crate ships a one-pixel host smoke
//! test that asserts the shader's output matches the GPU bit-for-bit.

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{USizeVec3, UVec4};
use spirv_std::arch::opencl_std as ocl;
use spirv_std::cl::{Float3, Int2};
// `num_traits::Float` is needed on SPIR-V targets to bring `cos`/`sin`/
// `powf`/`exp` into scope on bare `f32` (no `std`), where the libm
// intercept rewrites them to `OpExtInst <OpenCL.std> {cos, sin, …}`.
// On the host the same methods come from `std`, so the import is unused
// — `#[cfg(target_arch = "spirv")]` keeps clippy quiet on the host
// runner build.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{Image, glam, spirv};

// ── Numeric tolerances ─────────────────────────────────────────
const EPSILON: f32 = 0.001; // small-distance tolerance reused throughout

// ── Scene ──────────────────────────────────────────────────────
const SPHERE_A: Float3 = Float3::new(-0.7, 0.0, 0.0);
const SPHERE_B: Float3 = Float3::new(0.6, -0.2, 0.0);
const RADIUS_A: f32 = 0.9;
const RADIUS_B: f32 = 0.7;
const GROUND_Y: f32 = -0.9;
const SMIN_K: f32 = 0.45; // smooth-min blend radius between the two SDFs
// Bias when picking the ground vs. blob palette: hit just above the plane
// still counts as ground (avoids flicker on grazing hits).
const GROUND_BIAS: f32 = 0.01;

// ── Primary ray march ──────────────────────────────────────────
const MAX_STEPS: u32 = 96;
const MAX_DIST: f32 = 30.0;

// ── Soft-shadow march ──────────────────────────────────────────
const SHADOW_STEPS: u32 = 32;
const SHADOW_MAX_DIST: f32 = 12.0;
const SHADOW_T_START: f32 = 0.02; // step away from the surface to dodge self-shadow
const SHADOW_K: f32 = 8.0; // penumbra width (smaller = softer)
const SHADOW_STEP_MIN: f32 = 0.05; // clamp the per-step distance
const SHADOW_STEP_MAX: f32 = 0.5;

// ── Camera ─────────────────────────────────────────────────────
const CAM_RO: Float3 = Float3::new(3.0, 1.6, 4.0);
const CAM_TARGET: Float3 = Float3::ZERO;
const CAM_WORLD_UP: Float3 = Float3::Y;
const FOV_SCALE: f32 = 0.7;

// ── Sun & shading ──────────────────────────────────────────────
const SUN_AZ: f32 = 0.7;
const SUN_EL: f32 = 0.6;
const SUN_COLOR: Float3 = Float3::new(1.0, 0.95, 0.85);
const AMBIENT: f32 = 0.15;
const DIFFUSE: f32 = 0.7;
const SPECULAR_POWER: f32 = 32.0;
const FOG_DENSITY: f32 = 0.06;

// ── Sky gradient ───────────────────────────────────────────────
const SKY_ZENITH: Float3 = Float3::new(0.30, 0.55, 0.85);
const SKY_BAND: Float3 = Float3::new(0.85, 0.78, 0.62);
const SKY_HORIZON_LO: f32 = -0.05;
const SKY_HORIZON_HI: f32 = 0.45;

// ── Surface palette ────────────────────────────────────────────
const COLOR_GROUND: Float3 = Float3::new(0.55, 0.55, 0.60);
const COLOR_BLOB: Float3 = Float3::new(0.85, 0.55, 0.40);

fn ray_at(ro: Float3, rd: Float3, t: f32) -> Float3 {
    rd.mul_add(Float3::splat(t), ro)
}

// Polynomial smooth-min — blends two SDFs over a radius `k`. The
// scalar `clamp` stays as `ocl::clamp` because `f32::clamp` from
// `core` inlines as branchy Rust source on SPIR-V (no fast-path),
// while `ocl::clamp` lowers to a single `OpExtInst Fclamp`. Same for
// `mix` — there's no scalar `f32::mix` in `core` to reach for.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = ocl::clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    ocl::mix(b, a, h) - k * h * (1.0 - h)
}

fn scene_sdf(p: Float3) -> f32 {
    let d_a = p.distance(SPHERE_A) - RADIUS_A;
    let d_b = p.distance(SPHERE_B) - RADIUS_B;
    let blob = smin(d_a, d_b, SMIN_K);
    let plane = p.y() - GROUND_Y;
    blob.min(plane)
}

fn scene_normal(p: Float3) -> Float3 {
    let ex = Float3::new(EPSILON, 0.0, 0.0);
    let ey = Float3::new(0.0, EPSILON, 0.0);
    let ez = Float3::new(0.0, 0.0, EPSILON);
    let dx = scene_sdf(p + ex) - scene_sdf(p - ex);
    let dy = scene_sdf(p + ey) - scene_sdf(p - ey);
    let dz = scene_sdf(p + ez) - scene_sdf(p - ez);
    Float3::new(dx, dy, dz).normalize()
}

fn march(ro: Float3, rd: Float3) -> (bool, f32) {
    let mut t = 0.0f32;
    let mut i = 0u32;
    while i < MAX_STEPS {
        let d = scene_sdf(ray_at(ro, rd, t));
        if d < EPSILON {
            return (true, t);
        }
        t += d;
        if t > MAX_DIST {
            break;
        }
        i += 1;
    }
    (false, t)
}

// Cone-marched soft shadow. `SHADOW_K` controls penumbra width.
fn soft_shadow(ro: Float3, rd: Float3) -> f32 {
    let mut t = SHADOW_T_START;
    let mut res = 1.0f32;
    let mut i = 0u32;
    while i < SHADOW_STEPS {
        let d = scene_sdf(ray_at(ro, rd, t));
        if d < EPSILON {
            return 0.0;
        }
        res = res.min(SHADOW_K * d / t);
        t += ocl::clamp(d, SHADOW_STEP_MIN, SHADOW_STEP_MAX);
        if t > SHADOW_MAX_DIST {
            break;
        }
        i += 1;
    }
    ocl::clamp(res, 0.0, 1.0)
}

fn sky(rd: Float3) -> Float3 {
    let h = ocl::smoothstep(SKY_HORIZON_LO, SKY_HORIZON_HI, rd.y());
    SKY_BAND.lerp(SKY_ZENITH, h)
}

fn shade(p: Float3, n: Float3, ro: Float3, sun: Float3, base: Float3) -> Float3 {
    let view = (ro - p).normalize();
    let ndotl = ocl::clamp(n.dot(sun), 0.0, 1.0);
    let shadow = soft_shadow(p, sun);

    // Phong specular: reflect view about normal, dot with sun, raise to power.
    let refl = n * (2.0 * view.dot(n)) - view;
    let spec = ocl::clamp(refl.dot(sun), 0.0, 1.0).powf(SPECULAR_POWER) * shadow;

    let diff = DIFFUSE * ndotl * shadow;
    base * (SUN_COLOR * diff + Float3::splat(AMBIENT)) + SUN_COLOR * spec
}

/// Computes the colour at a single pixel. Factored out of the kernel
/// so the host smoke test can call it with the same code path.
pub fn pixel_color(u: f32, v: f32) -> Float3 {
    // Camera basis.
    let forward = (CAM_TARGET - CAM_RO).normalize();
    let right = forward.cross(CAM_WORLD_UP).normalize();
    let cam_up = right.cross(forward);
    let rd = (forward + (right * (u * FOV_SCALE)) + (cam_up * (v * FOV_SCALE))).normalize();

    // Sun direction from spherical coords (azimuth, elevation).
    // `f32::sin`/`cos` lower to `OpExtInst OpenCL.std {sin,cos}` via
    // the libm intercept on Kernel targets — same codegen as
    // `ocl::sin`/`cos`, just nicer to read.
    let sun = Float3::new(
        SUN_EL.cos() * SUN_AZ.sin(),
        SUN_EL.sin(),
        SUN_EL.cos() * SUN_AZ.cos(),
    )
    .normalize();

    let (hit, t) = march(CAM_RO, rd);

    if hit {
        let p = ray_at(CAM_RO, rd, t);
        let n = scene_normal(p);
        let base = if p.y() < GROUND_Y + GROUND_BIAS {
            COLOR_GROUND
        } else {
            COLOR_BLOB
        };
        let surf = shade(p, n, CAM_RO, sun, base);
        // Distance fog: blend toward sky as t grows.
        let fog = (-t * FOG_DENSITY).exp();
        sky(rd).lerp(surf, fog)
    } else {
        sky(rd)
    }
}

#[spirv(kernel)]
pub fn raymarch(
    #[spirv(global_invocation_id)] id: USizeVec3,
    image: &mut Image!(2D, type=u32, sampled=false),
    width: u32,
    height: u32,
) {
    let px = id.x as u32;
    let py = id.y as u32;
    if px >= width || py >= height {
        return;
    }

    // Pixel → NDC, with y flipped so origin is top-left.
    let aspect = width as f32 / height as f32;
    let u = (2.0 * (px as f32 + 0.5) / width as f32 - 1.0) * aspect;
    let v = 1.0 - 2.0 * (py as f32 + 0.5) / height as f32;

    let color = pixel_color(u, v);

    // Convert the saturated colour to per-channel u32 + alpha, then to
    // `glam::UVec4` for the image write — the storage-image `texels` arg
    // expects glam's vector type for the pixel format.
    let rgb = (color.clamp(Float3::ZERO, Float3::ONE) * 255.0).as_uint3();
    let rgba = rgb.extend(255);
    let out = UVec4::from_array(rgba.to_array());

    let coord = Int2::new(px as i32, py as i32);
    unsafe {
        image.write(coord, out);
    }
}
