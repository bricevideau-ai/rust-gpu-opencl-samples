#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{IVec2, USizeVec3, Vec3};
use spirv_std::arch::opencl_std as ocl;
use spirv_std::{Image, glam, spirv};

// ── Numeric tolerances ─────────────────────────────────────────
const EPSILON: f32 = 0.001; // small-distance tolerance reused throughout

// ── Scene ──────────────────────────────────────────────────────
const SPHERE_A: Vec3 = Vec3::new(-0.7, 0.0, 0.0);
const SPHERE_B: Vec3 = Vec3::new(0.6, -0.2, 0.0);
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
const CAM_RO: Vec3 = Vec3::new(3.0, 1.6, 4.0);
const CAM_TARGET: Vec3 = Vec3::ZERO;
const CAM_WORLD_UP: Vec3 = Vec3::Y;
const FOV_SCALE: f32 = 0.7;

// ── Sun & shading ──────────────────────────────────────────────
const SUN_AZ: f32 = 0.7;
const SUN_EL: f32 = 0.6;
const SUN_COLOR: Vec3 = Vec3::new(1.0, 0.95, 0.85);
const AMBIENT: f32 = 0.15;
const DIFFUSE: f32 = 0.7;
const SPECULAR_POWER: f32 = 32.0;
const FOG_DENSITY: f32 = 0.06;

// ── Sky gradient ───────────────────────────────────────────────
const SKY_ZENITH: Vec3 = Vec3::new(0.30, 0.55, 0.85);
const SKY_BAND: Vec3 = Vec3::new(0.85, 0.78, 0.62);
const SKY_HORIZON_LO: f32 = -0.05;
const SKY_HORIZON_HI: f32 = 0.45;

// ── Surface palette ────────────────────────────────────────────
const COLOR_GROUND: Vec3 = Vec3::new(0.55, 0.55, 0.60);
const COLOR_BLOB: Vec3 = Vec3::new(0.85, 0.55, 0.40);

fn ray_at(ro: Vec3, rd: Vec3, t: f32) -> Vec3 {
    ocl::fma(rd, Vec3::splat(t), ro)
}

// Polynomial smooth-min — blends two SDFs over a radius `k`.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = ocl::clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    ocl::mix(b, a, h) - k * h * (1.0 - h)
}

fn scene_sdf(p: Vec3) -> f32 {
    let d_a = ocl::distance(p, SPHERE_A) - RADIUS_A;
    let d_b = ocl::distance(p, SPHERE_B) - RADIUS_B;
    let blob = smin(d_a, d_b, SMIN_K);
    let plane = p.y - GROUND_Y;
    ocl::fmin(blob, plane)
}

fn scene_normal(p: Vec3) -> Vec3 {
    let ex = Vec3::new(EPSILON, 0.0, 0.0);
    let ey = Vec3::new(0.0, EPSILON, 0.0);
    let ez = Vec3::new(0.0, 0.0, EPSILON);
    let dx = scene_sdf(p + ex) - scene_sdf(p - ex);
    let dy = scene_sdf(p + ey) - scene_sdf(p - ey);
    let dz = scene_sdf(p + ez) - scene_sdf(p - ez);
    ocl::normalize(Vec3::new(dx, dy, dz))
}

fn march(ro: Vec3, rd: Vec3) -> (bool, f32) {
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
fn soft_shadow(ro: Vec3, rd: Vec3) -> f32 {
    let mut t = SHADOW_T_START;
    let mut res = 1.0f32;
    let mut i = 0u32;
    while i < SHADOW_STEPS {
        let d = scene_sdf(ray_at(ro, rd, t));
        if d < EPSILON {
            return 0.0;
        }
        res = ocl::fmin(res, SHADOW_K * d / t);
        t += ocl::clamp(d, SHADOW_STEP_MIN, SHADOW_STEP_MAX);
        if t > SHADOW_MAX_DIST {
            break;
        }
        i += 1;
    }
    ocl::clamp(res, 0.0, 1.0)
}

fn sky(rd: Vec3) -> Vec3 {
    let h = ocl::smoothstep(SKY_HORIZON_LO, SKY_HORIZON_HI, rd.y);
    ocl::mix(SKY_BAND, SKY_ZENITH, Vec3::splat(h))
}

fn shade(p: Vec3, n: Vec3, ro: Vec3, sun: Vec3, base: Vec3) -> Vec3 {
    let view = ocl::normalize(ro - p);
    let ndotl = ocl::clamp(n.dot(sun), 0.0, 1.0);
    let shadow = soft_shadow(p, sun);

    // Phong specular: reflect view about normal, dot with sun, raise to power.
    let refl = n * (2.0 * view.dot(n)) - view;
    let spec = ocl::pow(ocl::clamp(refl.dot(sun), 0.0, 1.0), SPECULAR_POWER) * shadow;

    let diff = DIFFUSE * ndotl * shadow;
    base * (SUN_COLOR * diff + Vec3::splat(AMBIENT)) + SUN_COLOR * spec
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

    // Camera basis.
    let forward = ocl::normalize(CAM_TARGET - CAM_RO);
    let right = ocl::normalize(forward.cross(CAM_WORLD_UP));
    let cam_up = right.cross(forward);
    let rd = ocl::normalize(forward + (right * (u * FOV_SCALE)) + (cam_up * (v * FOV_SCALE)));

    // Sun direction from spherical coords (azimuth, elevation).
    let sun = ocl::normalize(Vec3::new(
        ocl::cos(SUN_EL) * ocl::sin(SUN_AZ),
        ocl::sin(SUN_EL),
        ocl::cos(SUN_EL) * ocl::cos(SUN_AZ),
    ));

    let (hit, t) = march(CAM_RO, rd);

    let color = if hit {
        let p = ray_at(CAM_RO, rd, t);
        let n = scene_normal(p);
        let base = if p.y < GROUND_Y + GROUND_BIAS {
            COLOR_GROUND
        } else {
            COLOR_BLOB
        };
        let surf = shade(p, n, CAM_RO, sun, base);
        // Distance fog: blend toward sky as t grows.
        let fog = ocl::exp(-t * FOG_DENSITY);
        ocl::mix(sky(rd), surf, Vec3::splat(fog))
    } else {
        sky(rd)
    };

    let out = (ocl::clamp(color, Vec3::ZERO, Vec3::ONE) * 255.0)
        .as_uvec3()
        .extend(255);

    let coord = IVec2::new(px as i32, py as i32);
    unsafe {
        image.write(coord, out);
    }
}
