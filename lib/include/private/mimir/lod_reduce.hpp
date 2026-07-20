#pragma once

// LodReduce: standalone custom-CUDA implementation of the LOD grid reduction (see lod.hpp for the
// Vulkan atomic-scatter equivalent this replaces on datacenter GPUs, where the compute-shader path
// is ~190x slower than native CUDA). This header must NOT pull in any CUDA headers -- it is
// included from plain C++ translation units (lod.cpp, engine.cpp) compiled by the host compiler,
// not nvcc. All CUDA machinery lives behind the pImpl in lod_reduce.cu.
//
// Custom kernels only (NO CUB/Thrust): the scatter loop indexes particles with size_t/uint64_t so
// it scales past 2^32 particles, which CUB/Thrust's int-sized item counts cannot.
//
// This module is standalone and not wired into any render path yet (see
// .superpowers/sdd/task-1-brief.md, Task 1 of the LOD custom-CUDA rewrite).

#include <cstddef>
#include <cstdint>

// cudaStream_t is `typedef struct CUstream_st* cudaStream_t;` in <driver_types.h>. Forward-declare
// the identical type here (a repeated typedef of the same type is legal in C++, whether this
// header is included alone or alongside <cuda_runtime.h> in lod_reduce.cu) so callers can pass a
// real stream through this header without it depending on the CUDA toolkit.
struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace mimir
{

// Bins `count` particle positions into an N^3 grid over the fixed [-1,1]^3 domain (matching
// pathtrace_lod_scatter.slang's mapping exactly: cell index per axis =
// clamp(int((p+1)*0.5*N), 0, N-1), linear id = cx + N*(cy + N*cz)) and compacts one representative
// position per OCCUPIED cell: the mass centroid (centroid=true) or the cell's geometric center
// (centroid=false, = -1 + (cell_xyz + 0.5) * (2/N)). Implemented with a custom atomic
// scatter-into-an-N^3-accumulator + compacting emit pass (no sort): centroid placement uses
// atomics (atomicAdd on a u32 count and 3x int64 fixed-point position sums) so per-cell sums are
// order-independent and therefore deterministic run-to-run; cell-center placement uses a benign
// non-atomic occupancy write (every writer stores the identical value 1u). Either way the OCCUPIED
// SET is deterministic (a cell is occupied iff >=1 particle maps to it), independent of particle
// order.
class LodReduce
{
public:
    // gridN = cells/axis (cells = gridN^3; caller must keep gridN <= 1625 so cells < 2^32 and cell
    // ids fit a uint32 key). centroid: true -> mass centroid placement, false -> cell-center
    // placement. max_particles is an upper bound on `count` passed to reduce(); it may exceed
    // 2^32 (the scatter loop indexes with size_t/uint64_t regardless).
    LodReduce(uint64_t max_particles, uint32_t gridN, bool centroid);
    ~LodReduce();
    LodReduce(const LodReduce&) = delete;
    LodReduce& operator=(const LodReduce&) = delete;

    // Size (bytes) of the N^3 accumulator this instance holds: counts[N^3] (u32) plus, when
    // centroid, sums[3*N^3] (u64 fixed-point). Static so the caller can size/account for it (e.g.
    // VRAM-fit / logging) before constructing; mirrors the ctor's internal sizing exactly.
    static size_t accumulatorBytes(uint32_t gridN, bool centroid);

    // Reduce `count` positions (device ptr, packed float3, stride 12 B; count <= max_particles) on
    // `stream`: clears the N^3 accumulator, scatters every particle into it (size_t grid-stride
    // loop, so `count` may exceed 2^32), then emits one representative per occupied cell
    // (compacted, float3) into reduced_pos_dev. Returns the occupied-cell count via *occupied_dev
    // (a single uint32 in device memory). Entirely async on `stream` -- no host readback/sync
    // inside this call; the caller synchronizes (or otherwise orders against its own timeline)
    // before reading either output (e.g. before Vulkan reads reduced_pos_dev).
    void reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                void* reduced_pos_dev, uint32_t* occupied_dev);

private:
    struct Impl;
    Impl* impl;
};

} // namespace mimir
