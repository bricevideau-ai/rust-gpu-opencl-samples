//! Direct (O(N²)) n-body gravitational simulation using native OpenCL
//! `double3` vectors (`spirv_std::cl::Double3`) for all arithmetic.
//! Each kernel invocation advances the simulation by one timestep
//! using the leapfrog (kick-drift-kick) integrator.
//!
//! Exercises codegen paths that no other sample currently uses:
//!
//! - **`cl::Double3` end-to-end** in a hot loop. Every body's
//!   acceleration is computed from N pairwise interactions, each doing
//!   subtract / dot / scale on a native OpenCL 3-wide f64 vector with
//!   the canonical `double3` ABI (32-byte aligned, 32-byte size with
//!   trailing padding) — same layout host-side and device-side, no
//!   wrapper newtype required.
//! - **Array of structs** (`&mut [Body]`) as a `cross_workgroup`
//!   parameter, where the struct contains `Double3` fields (position +
//!   velocity). Exercises `OpAccessChain` with chained indices on a
//!   struct whose fields are themselves vectors.
//! - **`opencl_std::sqrt` on a scalar `f64`** in a tight loop (~N calls
//!   per thread).

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::U64Vec3;
use spirv_std::arch::opencl_std as ocl;
use spirv_std::cl::Double3;
use spirv_std::{glam, spirv};

/// Newtonian gravitational constant scaled for the sample's units. Real
/// G is ~6.674e-11; we use a unit value so position/velocity stay in
/// well-behaved ranges for visual debugging without rescaling.
const G: f64 = 1.0;

/// Plummer softening squared. Prevents the 1/r² force from blowing up
/// when two bodies pass close to each other, which would otherwise
/// require a much smaller timestep to remain stable.
const SOFTENING_SQ: f64 = 0.01;

/// Body state. `pos` and `vel` are native `cl::Double3` — same 32-byte
/// layout host- and device-side, no `OclDouble3`-style wrapper needed.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Body {
    pub pos: Double3,
    pub vel: Double3,
    pub mass: f64,
}

// Layout invariant — `Double3` is 32-byte aligned (vec3-double = vec4-double
// in the OpenCL ABI), so `Body` ends up 96 bytes with trailing padding.
const _: () = {
    assert!(core::mem::size_of::<Body>() == 96);
    assert!(core::mem::align_of::<Body>() == 32);
};

#[cfg(not(target_arch = "spirv"))]
impl Body {
    /// Constructs a body from raw `[x, y, z]` arrays — convenient when
    /// host code wants to spell out coordinates as literals.
    pub fn new(pos: [f64; 3], vel: [f64; 3], mass: f64) -> Self {
        Self {
            pos: Double3::from_array(pos),
            vel: Double3::from_array(vel),
            mass,
        }
    }

    /// A body at rest at the given position.
    pub fn at_rest(pos: [f64; 3], mass: f64) -> Self {
        Self::new(pos, [0.0; 3], mass)
    }
}

/// One leapfrog (kick-drift-kick) step:
///   - kick:   v(t+dt/2) = v(t)      + a(t)      · dt/2
///   - drift:  x(t+dt)   = x(t)      + v(t+dt/2) · dt
///   - kick:   v(t+dt)   = v(t+dt/2) + a(t+dt)   · dt/2
///
/// We compute both accelerations from the current `bodies` array and
/// write the new state into `next`. The host swaps the buffers between
/// frames.
#[spirv(kernel)]
pub fn step(
    #[spirv(global_invocation_id)] gid: U64Vec3,
    #[spirv(cross_workgroup)] bodies: &[Body],
    #[spirv(cross_workgroup)] next: &mut [Body],
    dt: f64,
) {
    let i = gid.x as usize;
    if i >= bodies.len() {
        return;
    }

    let me = bodies[i];

    // a(t) — acceleration from current positions.
    let a0 = accel(bodies, i, me.pos);

    // Half-kick + drift to predicted position.
    let half_dt = dt * 0.5;
    let v_half = me.vel + a0 * half_dt;
    let new_pos = me.pos + v_half * dt;

    // a(t + dt) — acceleration at predicted position. Uses the OLD
    // positions of the OTHER bodies; this is fine because all threads
    // are reading from `bodies` which is not being written to in this
    // kernel.
    let a1 = accel(bodies, i, new_pos);

    let new_vel = v_half + a1 * half_dt;

    next[i] = Body {
        pos: new_pos,
        vel: new_vel,
        mass: me.mass,
    };
}

/// Total acceleration on body `i` at position `pos`, summed over all
/// other bodies. Plummer-softened. All math is `Double3`-based; the
/// `*` of vector-by-scalar lowers to `OpVectorTimesScalar`.
#[inline]
fn accel(bodies: &[Body], i: usize, pos: Double3) -> Double3 {
    let mut a = Double3::default();
    let n = bodies.len();
    let mut j = 0usize;
    while j < n {
        if j != i {
            let other = bodies[j];
            let d = other.pos - pos;
            // |d|² + ε² (the softening avoids the 1/r blow-up at close
            // approach). `Double3::dot` is a paper-thin wrapper around
            // `ocl::dot`, both lower to core SPIR-V `OpDot`.
            let dist_sq = d.dot(d) + SOFTENING_SQ;
            // a += G * m_other * d / |d|³  =  G * m_other * d / dist_sq^(3/2)
            let inv_dist = ocl::rsqrt(dist_sq);
            let inv_dist3 = inv_dist * inv_dist * inv_dist;
            let s = G * other.mass * inv_dist3;
            a += d * s;
        }
        j += 1;
    }
    a
}
