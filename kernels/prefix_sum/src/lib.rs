#![cfg_attr(target_arch = "spirv", no_std)]

use glam::USizeVec3;
use spirv_std::arch::{
    group_exclusive_i_add, group_i_add, workgroup_memory_barrier_with_group_sync,
};
use spirv_std::{glam, spirv};

const WG_SIZE: usize = 32;

/// Workgroup-level exclusive prefix sum using subgroup operations and shared memory.
///
/// Algorithm:
///   1. Each work item loads one element from `input` into shared memory.
///   2. Barrier — ensures all loads are visible.
///   3. Subgroup exclusive scan (`group_exclusive_i_add`) computes the prefix sum
///      across all work items in the subgroup. On many OpenCL implementations
///      (including pocl), the subgroup spans the entire workgroup, so one call
///      produces the full workgroup-level prefix sum.
///   4. Subgroup reduce (`group_i_add`) computes the workgroup total.
///   5. Work item 0 stores the total in shared memory for verification.
///   6. Barrier — ensures the total is visible.
///   7. Each work item writes its prefix sum to `output`. If a `totals` buffer
///      is provided, work item 0 writes the workgroup total there.
///
/// NOTE: Using subgroup builtins (`subgroup_id`, `num_subgroups`) together with
/// subgroup ops and workgroup memory causes a crash on pocl 6.0 CPU. This kernel
/// avoids that combination. On devices where subgroup_size < workgroup_size, a
/// multi-level approach with cross-subgroup propagation would be needed.
#[spirv(kernel(threads(32)))]
pub fn prefix_sum_kernel(
    #[spirv(global_invocation_id)] global_id: USizeVec3,
    #[spirv(local_invocation_id)] local_id: USizeVec3,
    #[spirv(cross_workgroup)] input: &[u32],
    #[spirv(cross_workgroup)] output: &mut [u32],
    #[spirv(cross_workgroup)] totals: &mut [u32],
    #[spirv(workgroup)] shared: &mut [u32; WG_SIZE],
) {
    let lid = local_id.x as usize;
    let gid = global_id.x;

    // Load input into shared memory.
    shared[lid] = input[gid];

    workgroup_memory_barrier_with_group_sync();

    // Subgroup exclusive scan — computes prefix sum within the subgroup.
    let scanned = group_exclusive_i_add(shared[lid]);

    // Subgroup reduce — total of all elements in the subgroup.
    let total = group_i_add(shared[lid]);

    // Work item 0 stores the workgroup total in shared memory.
    if lid == 0 {
        shared[0] = total;
    }

    workgroup_memory_barrier_with_group_sync();

    // Write prefix sum to output.
    output[gid] = scanned;

    // Work item 0 writes the workgroup total to the totals buffer.
    if lid == 0 {
        let wg_id = global_id.x / WG_SIZE;
        totals[wg_id] = shared[0];
    }
}
