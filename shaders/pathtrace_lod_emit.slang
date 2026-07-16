// LOD grid-aggregation, emit/compact pass. One thread per cell (N^3 total). Occupied cells append
// one representative POSITION into a compacted reduced-position list via a global atomic counter. The
// representative is the per-cell mass centroid (sum(positions)/count) when centroid placement is
// active, else the cell's geometric centre. The slot order is nondeterministic but the count and the
// resulting reduced set are not (order-independent), so benchmark metrics stay reproducible.
//
// This pass writes centroid POSITIONS (float3), not BVH AABBs: the representative radius is a
// consumer concern now (the path-tracer's AABB writer, or a raster marker size), so the reduced set
// is render-mode-agnostic. See the transversal-LOD design.
//
// The per-cell count and fixed-point position sum are read through raw buffer-device-address (BDA)
// pointers (they exceed the descriptor cap at large N). Explicit 64-bit address arithmetic + cast
// (see pathtrace_aabbs.slang). Only the small global emit counter remains a descriptor.

// Fixed-point scale: maps [-1,1] -> [0, 2^30]. Matches the scatter pass and LOD_FIXEDPOINT_SCALE.
static const double SCALE = 1073741824.0; // 2^30

struct PushConstants
{
    float*    reducedPos; // out: compacted representative-position list, tightly packed x,y,z. BDA, offset 0.
    uint*     cellCounts; // in : per-cell uint occupancy (N^3). BDA.
    uint64_t* cellSums;   // in : per-cell fixed-point position sum, layout [3*lin + 0..2]. BDA.
                          //      Only dereferenced when centroid != 0 (may be 0/unallocated otherwise).
    uint  gridN;          // cells per axis
    uint  centroid;       // 0 = cell-center placement; != 0 = mass-centroid placement
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> globalCount; // in/out: element 0 = # emitted

[shader("compute")]
[numthreads(64, 1, 1)]
void emitMain(uint3 tid : SV_DispatchThreadID)
{
    uint lin   = tid.x;
    uint total = pc.gridN * pc.gridN * pc.gridN; // 512^3 = 134,217,728 fits in uint32
    if (lin >= total) { return; }

    // Per-cell occupancy via the BDA count pointer.
    uint c = *(uint*)(uint64_t(pc.cellCounts) + uint64_t(lin) * 4ull);
    if (c == 0u) { return; }

    float3 center;
    if (pc.centroid != 0u)
    {
        // centroid in [-1,1]: -1 + 2 * (sum/count)/SCALE, per axis.
        uint64_t base = uint64_t(pc.cellSums) + uint64_t(lin) * 24ull;
        uint64_t sx = *(uint64_t*)(base + 0ull);
        uint64_t sy = *(uint64_t*)(base + 8ull);
        uint64_t sz = *(uint64_t*)(base + 16ull);
        double inv = 1.0 / (double(c) * SCALE);
        center = float3(-1.0, -1.0, -1.0)
               + 2.0 * float3(float(double(sx) * inv), float(double(sy) * inv), float(double(sz) * inv));
    }
    else
    {
        uint gx = lin % pc.gridN;
        uint gy = (lin / pc.gridN) % pc.gridN;
        uint gz = lin / (pc.gridN * pc.gridN);
        float cs = 2.0 / float(pc.gridN);
        // Cell geometric centre in [-1,1]^3.
        center = float3(-1.0, -1.0, -1.0) + (float3(float(gx), float(gy), float(gz)) + 0.5) * cs;
    }

    uint slot;
    InterlockedAdd(globalCount[0], 1u, slot);

    // Append the representative position (float3, stride 12 B). Explicit 64-bit BDA address + cast
    // (see pathtrace_aabbs.slang): NEVER pointer-index, the NVIDIA compiler truncates the offset.
    *(float3*)(uint64_t(pc.reducedPos) + uint64_t(slot) * 12ull) = center;
}
