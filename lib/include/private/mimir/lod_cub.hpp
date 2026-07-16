#pragma once

// LodCub: standalone CUDA/CUB implementation of the LOD grid reduction (see lod.hpp for the
// Vulkan atomic-scatter equivalent this is meant to eventually replace/accelerate). This header
// must NOT pull in any CUDA/CUB headers -- it is included from plain C++ translation units
// (lod.cpp, engine.cpp) compiled by the host compiler, not nvcc. All CUB machinery lives behind
// the pImpl in lod_cub.cu.
//
// Task 1 of the CUB-LOD migration: this module is standalone and not wired into any render path
// yet (see .superpowers/sdd/cub-t1-brief.md).

#include <cstddef>
#include <cstdint>

// cudaStream_t is `typedef struct CUstream_st* cudaStream_t;` in <driver_types.h>. Forward-declare
// the identical type here (a repeated typedef of the same type is legal in C++, whether this
// header is included alone or alongside <cuda_runtime.h> in lod_cub.cu) so callers can pass a real
// stream through this header without it depending on the CUDA toolkit.
struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace mimir
{

// Bins `count` particle positions into an N^3 grid over the fixed [-1,1]^3 domain (matching
// pathtrace_lod_scatter.slang's mapping exactly: cell index per axis =
// clamp(int((p+1)*0.5*N), 0, N-1), linear id = cx + N*(cy + N*cz)) and compacts one representative
// position per OCCUPIED cell: the mass centroid (centroid=true) or the cell's geometric center
// (centroid=false, = -1 + (cell_xyz + 0.5) * (2/N)). Implemented with CUB radix-sort +
// run-length-encode/reduce-by-key (sort-based, order-independent), so the occupied-cell SET and
// centroids are deterministic run-to-run regardless of particle order -- unlike an atomic scatter.
class LodCub
{
public:
    // gridN = cells/axis (cells = gridN^3; caller must keep gridN <= 1625 so cells < 2^32 and cell
    // ids fit a uint32 key). centroid: true -> mass centroid placement, false -> cell-center
    // placement. max_particles is an upper bound on `count` passed to reduce() and sizes all
    // scratch up front; it may exceed 2^32 (the per-particle index widens to uint64 internally
    // when it does).
    LodCub(uint64_t max_particles, uint32_t gridN, bool centroid);
    ~LodCub();
    LodCub(const LodCub&) = delete;
    LodCub& operator=(const LodCub&) = delete;

    // Total device scratch this instance holds (bytes), for the VRAM-fit selection. Static so the
    // caller can decide BEFORE constructing (mirrors the ctor's internal sizing exactly).
    static size_t scratchBytes(uint64_t max_particles, uint32_t gridN, bool centroid);

    // Reduce `count` positions (device ptr, packed float3, stride 12 B; count <= max_particles) on
    // `stream`: writes the compacted representative positions (float3, stride 12 B) to
    // reduced_pos_dev, and returns the occupied-cell count via *occupied_dev (a single uint32 in
    // device memory). Entirely async on `stream` -- no host readback/sync inside this call; the
    // caller synchronizes (or otherwise orders against its own timeline) before reading either
    // output (e.g. before Vulkan reads reduced_pos_dev).
    void reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                void* reduced_pos_dev, uint32_t* occupied_dev);

private:
    struct Impl;
    Impl* impl;
};

} // namespace mimir
