#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace mimir {
// Reference max-pool of a fine N^3 int grid into a coarse M^3 grid (row-major x+N*(y+N*z)). Each coarse
// cell covers [c*N/M,(c+1)*N/M) per axis. NOT on the render path -- the voxel_lod vertex shader pools
// live; this mirrors that exact formula for tests and for an optional CA_LOD_CHECK. Requires 0 < M <= N.
void voxelPoolMax(const int* fine, uint32_t N, int* coarse, uint32_t M, cudaStream_t stream);

// Stream-compact the LIVING cells (state != 0) of a fine N^3 int grid (row-major x+N*(y+N*z)) into a
// list of their world-space centers, world = origin + (x,y,z)*spacing. Writes up to `capacity`
// positions (tightly packed float3) into `out_positions` and the total living count into the device
// counter `d_count` (zeroed by this call). NOT order-preserving. Used to feed the path tracer an
// O(living) AABB list instead of the O(N^3) dense grid. `d_count` must be a device uint32.
void voxelCompactLiving(const int* state, uint32_t N, float3 origin, float3 spacing,
    float* out_positions, uint32_t capacity, uint32_t* d_count, cudaStream_t stream);
}
