// LOD grid-aggregation, scatter pass. Bins each particle into an N^3 grid over the fixed
// [-1,1]^3 domain and atomically increments the per-cell occupancy count. When centroid placement
// is active it ALSO accumulates a per-cell fixed-point position sum (3 * int64) so the emit pass
// can place the representative sphere at the mass centroid instead of the cell centre. Integer
// atomics (count and the fixed-point sum) are order-independent => deterministic occupied-cell set
// and centroids run-to-run.
//
// All accumulators ride the push constants as raw buffer-device-address (BDA) pointers: the sum is
// up to 3 * N^3 * 8 B (>4 GiB at large N), past the maxStorageBufferRange descriptor cap, and the
// count moves to BDA alongside it. Explicit 64-bit address arithmetic + reinterpret, NEVER pointer
// indexing (NVIDIA truncates OpPtrAccessChain offsets past 4 GiB; see pathtrace_aabbs.slang). The
// int64 atomic-add form was validated in lod_atomic_spike.slang (Task 1).

// Fixed-point scale: maps [-1,1] -> [0, 2^30]. Matches RayTracingContext::LOD_FIXEDPOINT_SCALE.
static const double SCALE = 1073741824.0; // 2^30

struct PushConstants
{
    float*    positions;  // in : interop positions, tightly packed x,y,z (stride 12 B). BDA, offset 0.
    uint*     cellCounts; // out: per-cell uint occupancy (N^3). BDA.
    uint64_t* cellSums;   // out: per-cell fixed-point position sum, layout [3*lin + 0..2]. BDA.
                          //      Only dereferenced when centroid != 0 (may be 0/unallocated otherwise).
    uint64_t  count;      // 64-bit particle count
    uint      gridN;      // cells per axis
    uint      centroid;   // 0 = count only (cell-center placement); != 0 = also accumulate the sum
    uint      stride;     // total dispatched threads (grid-stride step); <= 2^31
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void scatterMain(uint3 tid : SV_DispatchThreadID)
{
    // Bounded dispatch + 64-bit grid-stride loop (see pathtrace_aabbs.slang): tid.x stays < 2^32,
    // each thread walks the full 64-bit particle range in `stride` steps. For count <= 2^32 the
    // stride >= count so every particle is binned exactly once. The count/sum atomics are
    // order-independent, so the occupied-cell set and centroids stay deterministic run-to-run.
    for (uint64_t i = tid.x; i < pc.count; i += uint64_t(pc.stride))
    {
        // Explicit 64-bit BDA address arithmetic + cast (NVIDIA truncates OpPtrAccessChain offsets to
        // 32 bits past 4 GiB; see pathtrace_aabbs.slang).
        uint64_t pos_addr = uint64_t(pc.positions) + i * 12ull;
        float px = *(float*)(pos_addr + 0);
        float py = *(float*)(pos_addr + 4);
        float pz = *(float*)(pos_addr + 8);

        float n = float(pc.gridN);
        // Fixed domain [-1,1] -> [0,1] -> cell index, clamped to [0, N-1].
        int cx = int(clamp((px + 1.0) * 0.5 * n, 0.0, n - 1.0));
        int cy = int(clamp((py + 1.0) * 0.5 * n, 0.0, n - 1.0));
        int cz = int(clamp((pz + 1.0) * 0.5 * n, 0.0, n - 1.0));
        uint lin = uint(cx) + pc.gridN * (uint(cy) + pc.gridN * uint(cz));

        uint64_t count_addr = uint64_t(pc.cellCounts) + uint64_t(lin) * 4ull;
        if (pc.centroid != 0u)
        {
            // Centroid placement needs the TRUE per-cell count (it is the denominator of the mass
            // average in the emit pass), so the count must be an atomic add.
            uint old32;
            InterlockedAdd(*(uint*)count_addr, 1u, old32);

            // Quantize each component of the clamped position in [-1,1] to [0, 2^30] fixed-point.
            uint64_t qx = uint64_t((clamp(double(px), -1.0, 1.0) + 1.0) * 0.5 * SCALE);
            uint64_t qy = uint64_t((clamp(double(py), -1.0, 1.0) + 1.0) * 0.5 * SCALE);
            uint64_t qz = uint64_t((clamp(double(pz), -1.0, 1.0) + 1.0) * 0.5 * SCALE);
            // Sum layout: 3 * uint64 per cell at [3*lin + 0..2] => base = ptr + lin*24, comps at +0/+8/+16.
            // 64-bit OpAtomicIAdd on PhysicalStorageBuffer (validated in lod_atomic_spike.slang).
            uint64_t base = uint64_t(pc.cellSums) + uint64_t(lin) * 24ull;
            uint64_t old64;
            InterlockedAdd(*(uint64_t*)(base + 0ull),  qx, old64);
            InterlockedAdd(*(uint64_t*)(base + 8ull),  qy, old64);
            InterlockedAdd(*(uint64_t*)(base + 16ull), qz, old64);
        }
        else
        {
            // Cell-center placement needs only OCCUPANCY -- the emit pass merely tests count != 0 and
            // places the representative at the cell's geometric center (never reads the count value).
            // So drop the atomic entirely: a plain, non-atomic store of 1 is a BENIGN data race --
            // every particle landing in this cell writes the identical value, so whichever store wins
            // the cell still reads 1. This turns the per-particle atomic into a plain memory store,
            // which is the dominant cost of the reduction at huge N.
            *(uint*)count_addr = 1u;
        }
    }
}
