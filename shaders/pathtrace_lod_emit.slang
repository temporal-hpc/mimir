// LOD grid-aggregation, emit/compact pass. One thread per cell (N^3 total). Occupied cells append
// one sphere AABB (at the cell's geometric centre, radius sized to the cell) into a compacted list
// via a global atomic counter. The slot order is nondeterministic but the count and the resulting
// BVH/image are not (order-independent), so benchmark metrics stay reproducible.

// Matches VkAabbPositionsKHR: 6 tightly packed floats (24 bytes), min xyz then max xyz.
struct Aabb { float minx, miny, minz, maxx, maxy, maxz; };

struct PushConstants
{
    Aabb* aabbs;  // out: compacted AABB list. BDA, offset 0.
    uint  gridN;  // cells per axis
    float radius; // sphere world radius (= coverage * cellSize / 2)
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> cellCounts;  // in : per-cell occupancy
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> globalCount; // in/out: element 0 = # emitted

[shader("compute")]
[numthreads(64, 1, 1)]
void emitMain(uint3 tid : SV_DispatchThreadID)
{
    uint lin   = tid.x;
    uint total = pc.gridN * pc.gridN * pc.gridN; // 512^3 = 134,217,728 fits in uint32
    if (lin >= total) { return; }
    if (cellCounts[lin] == 0u) { return; }

    uint gx = lin % pc.gridN;
    uint gy = (lin / pc.gridN) % pc.gridN;
    uint gz = lin / (pc.gridN * pc.gridN);
    float cs = 2.0 / float(pc.gridN);
    // Cell geometric centre in [-1,1]^3.
    float3 c = float3(-1.0, -1.0, -1.0) + (float3(float(gx), float(gy), float(gz)) + 0.5) * cs;

    uint slot;
    InterlockedAdd(globalCount[0], 1u, slot);

    Aabb a;
    a.minx = c.x - pc.radius; a.miny = c.y - pc.radius; a.minz = c.z - pc.radius;
    a.maxx = c.x + pc.radius; a.maxy = c.y + pc.radius; a.maxz = c.z + pc.radius;
    // Explicit 64-bit BDA address + cast (see pathtrace_aabbs.slang).
    *(Aabb*)(uint64_t(pc.aabbs) + uint64_t(slot) * 24ull) = a;
}
