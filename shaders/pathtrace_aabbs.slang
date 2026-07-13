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
    // Instance/AABB buffer as a raw buffer-device-address pointer (physical storage buffer), NOT a
    // bound storage-buffer descriptor: a descriptor is capped by maxStorageBufferRange (~2 GiB on
    // some drivers), which would refuse the multi-GiB AABB buffer at large N. Pointer first (offset
    // 0, 8-byte aligned).
    Aabb* aabbs;
    uint count;   // number of particles
    float radius; // sphere world radius (AABB half-extent)
};
[[vk::push_constant]] PushConstants pc;

// Interop positions as tightly packed floats (stride 12 B/particle: x,y,z), matching the CUDA
// layout. Bound as a float StructuredBuffer so the vec3 std430 padding never applies.
[[vk::binding(0, 0)]] StructuredBuffer<float> positions;

[shader("compute")]
[numthreads(64, 1, 1)]
void writeAabbsMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count) { return; }

    float px = positions[3 * i + 0];
    float py = positions[3 * i + 1];
    float pz = positions[3 * i + 2];
    float r  = pc.radius;

    Aabb a;
    a.minx = px - r; a.miny = py - r; a.minz = pz - r;
    a.maxx = px + r; a.maxy = py + r; a.maxz = pz + r;
    pc.aabbs[i] = a; // BDA store: pointer indexing, no descriptor-range limit
}
