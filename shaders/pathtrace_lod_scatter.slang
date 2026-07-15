// LOD grid-aggregation, scatter pass. Bins each particle into an N^3 grid over the fixed
// [-1,1]^3 domain and atomically increments the per-cell occupancy count. Count only: the
// representative sphere is placed at the cell centre by the emit pass, so no position sum is
// needed. Integer-count atomics are order-independent => deterministic occupied-cell set.

// Positions ride the push constants as a raw buffer-device-address pointer (the buffer is many
// GiB at large N; a descriptor is capped by maxStorageBufferRange). The per-cell count buffer is
// small (<= 537 MB at 512^3) so it is a bound RWStructuredBuffer that supports InterlockedAdd.
struct PushConstants
{
    float* positions; // in: interop positions, tightly packed x,y,z (stride 12 B). BDA, offset 0.
    uint   count;     // number of particles
    uint   gridN;     // cells per axis
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> cellCounts; // out: one uint per cell (N^3)

[shader("compute")]
[numthreads(64, 1, 1)]
void scatterMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count) { return; }

    // Explicit 64-bit BDA address arithmetic + cast (NVIDIA truncates OpPtrAccessChain offsets to
    // 32 bits past 4 GiB; see pathtrace_aabbs.slang).
    uint64_t pos_addr = uint64_t(pc.positions) + uint64_t(i) * 12ull;
    float px = *(float*)(pos_addr + 0);
    float py = *(float*)(pos_addr + 4);
    float pz = *(float*)(pos_addr + 8);

    float n = float(pc.gridN);
    // Fixed domain [-1,1] -> [0,1] -> cell index, clamped to [0, N-1].
    int cx = int(clamp((px + 1.0) * 0.5 * n, 0.0, n - 1.0));
    int cy = int(clamp((py + 1.0) * 0.5 * n, 0.0, n - 1.0));
    int cz = int(clamp((pz + 1.0) * 0.5 * n, 0.0, n - 1.0));
    uint lin = uint(cx) + pc.gridN * (uint(cy) + pc.gridN * uint(cz));

    uint old;
    InterlockedAdd(cellCounts[lin], 1u, old);
}
