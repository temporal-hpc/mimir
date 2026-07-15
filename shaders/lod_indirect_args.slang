// LOD indirect-args finalize pass. A single-thread compute dispatch that turns the GPU-resident
// occupied-cell count (from the LOD emit pass) into a ready-to-consume Vk*IndirectCommand, so the
// raster draw can source its varying count via vkCmdDraw*Indirect with NO host readback / stall.
//
// Mode-agnostic by construction: the host passes the byte offset of the command's VARYING field
// (the one that gets the count) and the value for the fixed instanceCount field as push constants.
//   - point modes  (none / phong, VkDrawIndirectCommand): varyingByteOffset = 0 (vertexCount),
//     fixedInstanceCount = 1 (one instance). firstVertex / firstInstance come pre-zeroed by the
//     host vkCmdFillBuffer.
// The count buffer is the same HOST_VISIBLE emit counter the LOD emit pass wrote (element 0).
//
// BDA note: like the rest of the LOD/AABB shaders, addresses are computed in explicit 64-bit integer
// arithmetic and cast -- NEVER pointer-indexed (the NVIDIA compiler truncates OpPtrAccessChain
// offsets to 32 bits). Here the offsets are tiny, but we keep the same discipline for consistency.

struct PushConstants
{
    uint* indirect;           // out: the Vk*IndirectCommand buffer (BDA, offset 0). Pre-filled by host.
    uint* count;              // in : occupied-cell count, element 0 (BDA). Same buffer as the emit counter.
    uint  varyingByteOffset;  // byte offset within the command of the field that receives count[0]
    uint  fixedInstanceCount; // value written to the instanceCount field (offset 4) for point modes
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void finalizeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) { return; }

    uint n = *(uint*)(uint64_t(pc.count) + 0ull);

    // Varying field (vertexCount for point modes) = occupied count.
    *(uint*)(uint64_t(pc.indirect) + uint64_t(pc.varyingByteOffset)) = n;
    // instanceCount field lives at byte offset 4 in VkDrawIndirectCommand.
    *(uint*)(uint64_t(pc.indirect) + 4ull) = pc.fixedInstanceCount;
}
