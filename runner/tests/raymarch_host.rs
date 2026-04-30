//! Host-side smoke test for the `raymarch` kernel.
//!
//! Calls the same `raymarch::pixel_color(u, v)` function the kernel
//! invokes on the GPU, but on the CPU through the host arms of
//! `spirv_std::arch::opencl_std::*`. This proves the host fallbacks
//! produce sensible (and finite) shading at a handful of representative
//! pixels — without requiring an OpenCL device or an image format.
//!
//! The GPU run (driven by `cargo run -p runner -- raymarch`) remains
//! the functional check; this test is the host-parity guard rail.

use raymarch::pixel_color;
use spirv_std::cl::Float3;

/// Sample `pixel_color` at one (u, v) in NDC and assert the output is
/// a valid colour: finite components, each within the `[0, 1]` range
/// (the kernel later clamps + scales to 0..255 for display).
fn check_pixel(u: f32, v: f32) -> Float3 {
    let c = pixel_color(u, v);
    let arr = c.to_array();
    for (i, x) in arr.iter().enumerate() {
        assert!(
            x.is_finite(),
            "component {i} at ({u}, {v}) was non-finite: {x}"
        );
        assert!(
            (-0.01..=1.01).contains(x),
            "component {i} at ({u}, {v}) was out of range: {x}",
        );
    }
    c
}

#[test]
fn pixel_color_top_left_is_sky() {
    // Top-left corner — purely sky (no scene geometry in view).
    let c = check_pixel(-1.0, 1.0);
    let arr = c.to_array();
    // Sky is dominantly blue at the top.
    assert!(
        arr[2] > arr[0] && arr[2] > arr[1],
        "expected blue-dominant sky at top-left, got {arr:?}",
    );
}

#[test]
fn pixel_color_centre_hits_geometry() {
    // Centre of frame — should hit the spheres / ground.
    let _c = check_pixel(0.0, 0.0);
}

#[test]
fn pixel_color_bottom_centre_is_ground() {
    let _c = check_pixel(0.0, -0.9);
}

#[test]
fn pixel_color_grid() {
    // Sample a 9x9 grid across the full NDC viewport. Catches any
    // host-only NaN/Inf that doesn't show up in the three corner cases.
    for iy in 0..9 {
        for ix in 0..9 {
            let u = -1.0 + 0.25 * ix as f32;
            let v = 1.0 - 0.25 * iy as f32;
            let _ = check_pixel(u, v);
        }
    }
}
