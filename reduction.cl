#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sum(__global double* a, __local float* b,
                  __global double* partials) {
    int gid = get_global_id(0);

    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int wg = get_group_id(0);

    // Copy to local memory
    b[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction within work group, sum is left in b[0]
    for (int stride=lsize>>1; stride>0; stride>>=1) {
        if (lid < stride) {
            b[lid] += b[lid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Local thread 0 copies its work group sum to the result array
    if (lid == 0) {
        partials[wg] = b[0];
    }
}


__kernel void sum_axis1(__global double* a, __local float* b,
                        __global double* partials) {
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int gsize1 = get_global_size(1);

    int lid = get_local_id(1);
    int lsize = get_local_size(1);
    int wg = get_group_id(1);


    // Copy row to local memory
    b[lid] = a[gid0*gsize1 + gid1];
    barrier(CLK_LOCAL_MEM_FENCE);
    printf("%d %d <- %f\n", gid0, gid1, b[lid]);

    // Reduction within work group, sum is left in b[0]
    for (int stride=lsize>>1; stride>0; stride>>=1) {
        if (lid < stride) {
            b[lid] += b[lid+stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Local thread 0 copies its work group sum to the result array
    if (lid == 0) {
        partials[gid0 + wg] = b[0];
    }
}
