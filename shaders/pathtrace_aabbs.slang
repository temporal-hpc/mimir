// AABB-writer compute shader (LightModel::PathTracing, BLAS-of-primitives). Reads the live interop
// position buffer in place and writes one VkAabbPositionsKHR per particle for the per-frame BLAS
// rebuild. Replaces the old per-instance TLAS writer: representing each particle as a procedural
// AABB primitive in a single BLAS (with a ray-sphere intersection shader) lifts the count cap from
// maxInstanceCount (~2^24, one TLAS instance per particle) to maxPrimitiveCount (~2^29).

// Matches VkAabbPositionsKHR: 6 tightly packed floats (24 bytes), min xyz then max xyz.
struct Aabb
{
    float minx, miny, minz;
    float maxx, maxy, maxz;
};

struct PushConstants
{
    // Both buffers ride the push constants as raw buffer-device-address pointers (physical storage
    // buffers), NOT bound storage-buffer descriptors: a descriptor is capped by maxStorageBufferRange
    // (~2 GiB on some drivers), which would refuse the multi-GiB AABB buffer -- and cap the readable
    // position range -- at large N. With BDA the only ceiling is maxPrimitiveCount / VRAM. Pointers
    // first (8-byte aligned).
    Aabb* aabbs;      // output: one AABB per particle
    float* positions; // input: interop positions, tightly packed x,y,z, shared w/ CUDA. Stride is
                       // 12 B (float3) unless double_positions is set, then 24 B (double3) --
                       // either way read here and cast down to float for the AABB.
    uint64_t count;   // 64-bit particle count
    float radius;     // sphere world radius (AABB half-extent)
    uint stride;      // total dispatched threads (grid-stride step); <= 2^31
    uint double_positions; // 0 = float3 positions (12 B/particle), 1 = double3 (24 B/particle)
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void writeAabbsMain(uint3 tid : SV_DispatchThreadID)
{
    // Bounded dispatch + 64-bit grid-stride loop: the host caps the dispatch at kMaxGroups so
    // gl_GlobalInvocationID.x (tid.x) stays < 2^32 and the stride fits a uint32; each thread walks
    // the full 64-bit particle range in `stride` steps. For count <= 2^32 the stride >= count so
    // every particle is visited exactly once (identical result, determinism preserved).
    //
    // BDA accesses must compute the byte address in explicit 64-bit integer arithmetic and cast,
    // NOT index the pointer (pc.aabbs[i] / pc.aabbs[uint64_t(i)]): pointer indexing lowers to
    // OpPtrAccessChain, whose index*stride offset the NVIDIA compiler truncates to 32 bits EVEN
    // WITH a 64-bit index, silently wrapping every access past 4 GiB into the buffer -- at stride
    // 24 that is only ~179M particles (stride 12: ~357M), and the wrapped stores land misaligned
    // across earlier records, shredding the AABB buffer. Verified on Turing/Ampere/Blackwell,
    // driver 610.43.02 (2026-07).
    for (uint64_t i = tid.x; i < pc.count; i += uint64_t(pc.stride))
    {
        float px, py, pz;
        if (pc.double_positions != 0)
        {
            // double3 positions: 24 B/particle, cast down to float only here at the AABB write --
            // VkAabbPositionsKHR is spec'd as float, so this narrowing is unavoidable regardless of
            // the simulation's own precision; the BVH/traversal cost is float-only either way.
            uint64_t pos_addr = uint64_t(pc.positions) + i * 24ull;
            px = float(*(double*)(pos_addr + 0));
            py = float(*(double*)(pos_addr + 8));
            pz = float(*(double*)(pos_addr + 16));
        }
        else
        {
            uint64_t pos_addr = uint64_t(pc.positions) + i * 12ull;
            px = *(float*)(pos_addr + 0);
            py = *(float*)(pos_addr + 4);
            pz = *(float*)(pos_addr + 8);
        }
        float r  = pc.radius;

        Aabb a;
        a.minx = px - r; a.miny = py - r; a.minz = pz - r;
        a.maxx = px + r; a.maxy = py + r; a.maxz = pz + r;
        *(Aabb*)(uint64_t(pc.aabbs) + i * 24ull) = a;
    }
}
