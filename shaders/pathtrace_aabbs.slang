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
    float* positions; // input: interop positions, tightly packed x,y,z (stride 12 B, shared w/ CUDA)
    uint count;       // number of particles
    float radius;     // sphere world radius (AABB half-extent)
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void writeAabbsMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count) { return; }

    float px = pc.positions[3 * i + 0];
    float py = pc.positions[3 * i + 1];
    float pz = pc.positions[3 * i + 2];
    float r  = pc.radius;

    Aabb a;
    a.minx = px - r; a.miny = py - r; a.minz = pz - r;
    a.maxx = px + r; a.maxy = py + r; a.maxz = pz + r;
    pc.aabbs[i] = a; // BDA store: pointer indexing, no descriptor-range limit
}
