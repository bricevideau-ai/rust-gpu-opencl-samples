#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

use glam::USizeVec3;
use spirv_std::arch::{group_i_add, workgroup_memory_barrier_with_group_sync};
use spirv_std::{glam, spirv};

const WG_SIZE: usize = 32;

/// Hierarchical reduction using subgroup ops and shared memory.
///
/// Algorithm:
///   1. Each work item copies its input element into shared memory.
///   2. Barrier.
///   3. Each subgroup reduces its portion via `group_i_add`.
///      The first item of each subgroup (`subgroup_local_id == 0`)
///      writes the partial sum to `shared[subgroup_id]`.
///   4. Barrier.
///   5. Work item 0 reduces the partial sums across subgroups.
///   6. Work item 0 writes the workgroup total to the output buffer.
#[spirv(kernel(threads(32)))]
pub fn reduce_kernel(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(subgroup_id)] subgroup_id: u32,
    #[spirv(subgroup_local_invocation_id)] subgroup_local_id: u32,
    #[spirv(num_subgroups)] num_subgroups: u32,
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; WG_SIZE],
) {
    let lid = local_id.x as u32;
    let gid = global_id.x;

    // Step 1: copy input to shared memory.
    shared[lid as usize] = input[gid];

    workgroup_memory_barrier_with_group_sync();

    // Step 2: subgroup reduce.
    let partial = group_i_add(shared[lid as usize]);

    // Step 3: first item of each subgroup writes partial sum.
    if subgroup_local_id == 0 {
        shared[subgroup_id as usize] = partial;
    }

    workgroup_memory_barrier_with_group_sync();

    // Step 4: work item 0 reduces across subgroups and writes output.
    if lid == 0 {
        let mut total = 0u32;
        let mut i = 0u32;
        while i < num_subgroups {
            total += shared[i as usize];
            i += 1;
        }
        let wg_id = gid / WG_SIZE;
        output[wg_id] = total;
    }
}
