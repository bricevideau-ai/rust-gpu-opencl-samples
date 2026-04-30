//! Direct (O(N²)) n-body gravitational simulation using `f64` vectors
//! (`glam::DVec3`) for all arithmetic. Each kernel invocation advances
//! the simulation by one timestep using the leapfrog (kick-drift-kick)
//! integrator.
//!
//! Exercises codegen paths that no other sample currently uses:
//!
//! - **`DVec3` (f64 vectors) end-to-end** in a hot loop. Every body's
//!   acceleration is computed from N pairwise interactions, each doing
//!   `DVec3` subtract / dot / scale. Validates the `Float64` capability
//!   together with the per-component vector codegen path.
//! - **Array of structs** (`&mut [Body]`) as a `cross_workgroup`
//!   parameter, where the struct contains `DVec3` fields (position +
//!   velocity). Exercises `OpAccessChain` with chained indices
//!   (`bodies[i].pos.x`) and the `MemberDecorate Offset` skip from
//!   commit 6, on a struct whose fields are themselves vectors.
//! - **`opencl_std::sqrt` on a scalar `f64`** in a tight loop (~N calls
//!   per thread). Most existing samples call math intrinsics only once
//!   or twice per kernel invocation.

#![cfg_attr(target_arch = "spirv", no_std)]

use glam::{DVec3, U64Vec3};
use spirv_std::arch::opencl_std as ocl;
use spirv_std::{glam, spirv};

/// Newtonian gravitational constant scaled for the sample's units. Real
/// G is ~6.674e-11; we use a unit value so position/velocity stay in
/// well-behaved ranges for visual debugging without rescaling.
const G: f64 = 1.0;

/// Plummer softening squared. Prevents the 1/r² force from blowing up
/// when two bodies pass close to each other, which would otherwise
/// require a much smaller timestep to remain stable.
const SOFTENING_SQ: f64 = 0.01;

/// Body state. `pos` and `vel` are `glam::DVec3` (f64 3-vectors).
///
/// Note: OpenCL gives `double3` the alignment of `double4` (32 bytes),
/// so the host-side struct needs explicit padding around `pos`/`vel`
/// and a 32-byte struct alignment. See the corresponding `Body` in the
/// runner.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Body {
    pub pos: DVec3,
    pub vel: DVec3,
    pub mass: f64,
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
/// other bodies. Plummer-softened. All math is `DVec3`-based.
#[inline]
fn accel(bodies: &[Body], i: usize, pos: DVec3) -> DVec3 {
    let mut a = DVec3::ZERO;
    let n = bodies.len();
    let mut j = 0usize;
    while j < n {
        if j != i {
            let other = bodies[j];
            let d = other.pos - pos;
            // |d|² + ε² (the softening avoids the 1/r blow-up at close
            // approach). `d.dot(d)` lowers to a single dot-product.
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
