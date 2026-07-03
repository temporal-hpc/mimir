// Instance-writer compute shader (LightModel::PathTracing, Phase 2). Reads the live interop
// position buffer in place and writes one VkAccelerationStructureInstanceKHR per particle for
// the per-frame TLAS rebuild. Keeps the zero-copy story: CUDA writes positions, this reads
// them without a copy. Runs on the graphics queue before the TLAS build (see engine barriers).

// Matches VkAccelerationStructureInstanceKHR (64 bytes): a row-major 3x4 transform (3 float4
// rows, translation in .w), then two packed uints, then the 64-bit BLAS reference.
struct Instance
{
    float4 row0; // (r, 0, 0, px)
    float4 row1; // (0, r, 0, py)
    float4 row2; // (0, 0, r, pz)
    uint instanceCustomIndex_mask; // customIndex:24 | mask:8
    uint sbtOffset_flags;          // sbtRecordOffset:24 | flags:8
    uint blas_ref_lo;              // accelerationStructureReference low 32 bits
    uint blas_ref_hi;              // accelerationStructureReference high 32 bits
};

struct PushConstants
{
    uint count;      // number of particles
    float radius;    // per-instance uniform scale (world radius)
    uint blas_ref_lo;
    uint blas_ref_hi;
    uint material_count; // # of SBT materials to cycle instances through (1 = all use material 0)
};
[[vk::push_constant]] PushConstants pc;

// Interop positions as tightly packed floats (stride 12 B/particle: x,y,z), matching the CUDA
// layout. Bound as a float StructuredBuffer so the vec3 std430 padding never applies.
[[vk::binding(0, 0)]] StructuredBuffer<float> positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<Instance> instances;

[shader("compute")]
[numthreads(64, 1, 1)]
void writeInstancesMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count) { return; }

    float px = positions[3 * i + 0];
    float py = positions[3 * i + 1];
    float pz = positions[3 * i + 2];
    float r  = pc.radius;

    Instance inst;
    inst.row0 = float4(r, 0.0, 0.0, px);
    inst.row1 = float4(0.0, r, 0.0, py);
    inst.row2 = float4(0.0, 0.0, r, pz);
    inst.instanceCustomIndex_mask = (i & 0xFFFFFFu) | (0xFFu << 24);       // mask = 0xFF
    // sbtRecordOffset selects this instance's material (SBT hit record). Cycle instances across the
    // registered materials; material_count == 1 makes every instance use material 0 (single material).
    uint mat = (pc.material_count > 1u) ? (i % pc.material_count) : 0u;
    inst.sbtOffset_flags          = (mat & 0xFFFFFFu)
        | (0x1u << 24);  // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE
    inst.blas_ref_lo              = pc.blas_ref_lo;
    inst.blas_ref_hi              = pc.blas_ref_hi;
    instances[i] = inst;
}
