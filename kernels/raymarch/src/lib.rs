#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{IVec2, USizeVec3, UVec4, Vec3};
use spirv_std::arch::opencl_std as ocl;
use spirv_std::{Image, glam, spirv};

const SPHERE_A: Vec3 = Vec3::new(-0.7, 0.0, 0.0);
const SPHERE_B: Vec3 = Vec3::new(0.6, -0.2, 0.0);
const RADIUS_A: f32 = 0.9;
const RADIUS_B: f32 = 0.7;
const GROUND_Y: f32 = -0.9;

const MAX_STEPS: u32 = 96;
const MAX_DIST: f32 = 30.0;
const SURF_EPS: f32 = 0.001;

fn length(v: Vec3) -> f32 {
    ocl::sqrt(v.dot(v))
}

fn normalize(v: Vec3) -> Vec3 {
    v * ocl::rsqrt(v.dot(v))
}

fn ray_at(ro: Vec3, rd: Vec3, t: f32) -> Vec3 {
    Vec3::new(
        ocl::fma(rd.x, t, ro.x),
        ocl::fma(rd.y, t, ro.y),
        ocl::fma(rd.z, t, ro.z),
    )
}

// Polynomial smooth-min — blends two SDFs over a radius `k`.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = ocl::clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    ocl::mix(b, a, h) - k * h * (1.0 - h)
}

fn scene_sdf(p: Vec3) -> f32 {
    let d_a = length(p - SPHERE_A) - RADIUS_A;
    let d_b = length(p - SPHERE_B) - RADIUS_B;
    let blob = smin(d_a, d_b, 0.45);
    let plane = p.y - GROUND_Y;
    ocl::fmin(blob, plane)
}

fn scene_normal(p: Vec3) -> Vec3 {
    let e = 0.001;
    let dx = scene_sdf(Vec3::new(p.x + e, p.y, p.z)) - scene_sdf(Vec3::new(p.x - e, p.y, p.z));
    let dy = scene_sdf(Vec3::new(p.x, p.y + e, p.z)) - scene_sdf(Vec3::new(p.x, p.y - e, p.z));
    let dz = scene_sdf(Vec3::new(p.x, p.y, p.z + e)) - scene_sdf(Vec3::new(p.x, p.y, p.z - e));
    normalize(Vec3::new(dx, dy, dz))
}

fn march(ro: Vec3, rd: Vec3) -> (bool, f32) {
    let mut t = 0.0f32;
    let mut i = 0u32;
    while i < MAX_STEPS {
        let d = scene_sdf(ray_at(ro, rd, t));
        if d < SURF_EPS {
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

// Cone-marched soft shadow. `k` controls penumbra width.
fn soft_shadow(ro: Vec3, rd: Vec3, k: f32) -> f32 {
    let mut t = 0.02f32;
    let mut res = 1.0f32;
    let mut i = 0u32;
    while i < 32 {
        let d = scene_sdf(ray_at(ro, rd, t));
        if d < 0.001 {
            return 0.0;
        }
        res = ocl::fmin(res, k * d / t);
        t += ocl::clamp(d, 0.05, 0.5);
        if t > 12.0 {
            break;
        }
        i += 1;
    }
    ocl::clamp(res, 0.0, 1.0)
}

fn sky(rd: Vec3) -> Vec3 {
    let h = ocl::smoothstep(-0.05, 0.45, rd.y);
    let zenith = Vec3::new(0.30, 0.55, 0.85);
    let band = Vec3::new(0.85, 0.78, 0.62);
    Vec3::new(
        ocl::mix(band.x, zenith.x, h),
        ocl::mix(band.y, zenith.y, h),
        ocl::mix(band.z, zenith.z, h),
    )
}

fn shade(p: Vec3, n: Vec3, ro: Vec3, sun: Vec3, sun_color: Vec3, base: Vec3) -> Vec3 {
    let view = normalize(ro - p);
    let ndotl = ocl::clamp(n.dot(sun), 0.0, 1.0);
    let shadow = soft_shadow(p, sun, 8.0);

    // Phong specular: reflect view about normal, dot with sun, raise to power.
    let refl = n * (2.0 * view.dot(n)) - view;
    let spec = ocl::pow(ocl::clamp(refl.dot(sun), 0.0, 1.0), 32.0) * shadow;

    let amb = 0.15;
    let diff = 0.7 * ndotl * shadow;
    Vec3::new(
        base.x * (amb + diff * sun_color.x) + spec * sun_color.x,
        base.y * (amb + diff * sun_color.y) + spec * sun_color.y,
        base.z * (amb + diff * sun_color.z) + spec * sun_color.z,
    )
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
    let ro = Vec3::new(3.0, 1.6, 4.0);
    let target = Vec3::new(0.0, 0.0, 0.0);
    let world_up = Vec3::new(0.0, 1.0, 0.0);
    let forward = normalize(target - ro);
    let right = normalize(forward.cross(world_up));
    let cam_up = right.cross(forward);

    let fov_scale = 0.7;
    let rd = normalize(forward + (right * (u * fov_scale)) + (cam_up * (v * fov_scale)));

    // Sun direction from spherical coords (azimuth, elevation).
    let sun_az = 0.7f32;
    let sun_el = 0.6f32;
    let sun = normalize(Vec3::new(
        ocl::cos(sun_el) * ocl::sin(sun_az),
        ocl::sin(sun_el),
        ocl::cos(sun_el) * ocl::cos(sun_az),
    ));
    let sun_color = Vec3::new(1.0, 0.95, 0.85);

    let (hit, t) = march(ro, rd);

    let color = if hit {
        let p = ray_at(ro, rd, t);
        let n = scene_normal(p);
        let base = if p.y < GROUND_Y + 0.01 {
            Vec3::new(0.55, 0.55, 0.60)
        } else {
            Vec3::new(0.85, 0.55, 0.40)
        };
        let surf = shade(p, n, ro, sun, sun_color, base);
        // Distance fog: blend toward sky as t grows.
        let fog = ocl::exp(-t * 0.06);
        let s = sky(rd);
        Vec3::new(
            ocl::mix(s.x, surf.x, fog),
            ocl::mix(s.y, surf.y, fog),
            ocl::mix(s.z, surf.z, fog),
        )
    } else {
        sky(rd)
    };

    let r = ocl::clamp(color.x, 0.0, 1.0);
    let g = ocl::clamp(color.y, 0.0, 1.0);
    let b = ocl::clamp(color.z, 0.0, 1.0);
    let out = UVec4::new(
        (r * 255.0) as u32,
        (g * 255.0) as u32,
        (b * 255.0) as u32,
        255,
    );

    let coord = IVec2::new(px as i32, py as i32);
    unsafe {
        image.write(coord, out);
    }
}
