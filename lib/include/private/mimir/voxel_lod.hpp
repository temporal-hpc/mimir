#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace mimir {
// Reference max-pool of a fine N^3 int grid into a coarse M^3 grid (row-major x+N*(y+N*z)). Each coarse
// cell covers [c*N/M,(c+1)*N/M) per axis. NOT on the render path -- the voxel_lod vertex shader pools
// live; this mirrors that exact formula for tests and for an optional CA_LOD_CHECK. Requires 0 < M <= N.
void voxelPoolMax(const int* fine, uint32_t N, int* coarse, uint32_t M, cudaStream_t stream);
}
