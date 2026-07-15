// LOD indirect-args finalize pass. A single-thread compute dispatch that turns the GPU-resident
// occupied-cell count (from the LOD emit pass) into a ready-to-consume Vk*IndirectCommand, so the
// raster draw can source its varying count via vkCmdDraw*Indirect with NO host readback / stall.
//
// Mode-agnostic by construction: the host passes ONLY the byte offset of the command's VARYING
// field (the one that gets the count). Every FIXED field (the icosphere indexCount / point
// instanceCount, plus firstVertex/firstIndex/vertexOffset/firstInstance = 0) is pre-filled by the
// host as a whole-command template (vkCmdUpdateBuffer) before this dispatch runs, so this shader
// touches exactly one field -- the varying count -- and needs no per-mode branching.
//   - point modes (none / phong, VkDrawIndirectCommand):        varyingByteOffset = 0 (vertexCount).
//   - mesh  mode  (phong-mesh, VkDrawIndexedIndirectCommand):   varyingByteOffset = 4 (instanceCount).
// The count buffer is the same HOST_VISIBLE emit counter the LOD emit pass wrote (element 0).
//
// BDA note: like the rest of the LOD/AABB shaders, addresses are computed in explicit 64-bit integer
// arithmetic and cast -- NEVER pointer-indexed (the NVIDIA compiler truncates OpPtrAccessChain
// offsets to 32 bits). Here the offsets are tiny, but we keep the same discipline for consistency.

struct PushConstants
{
    uint* indirect;           // out: the Vk*IndirectCommand buffer (BDA, offset 0). Template pre-filled by host.
    uint* count;              // in : occupied-cell count, element 0 (BDA). Same buffer as the emit counter.
    uint  varyingByteOffset;  // byte offset within the command of the field that receives count[0]
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void finalizeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u) { return; }

    uint n = *(uint*)(uint64_t(pc.count) + 0ull);

    // Write ONLY the varying field (vertexCount for points, instanceCount for mesh) = occupied count.
    // Every fixed field was pre-filled by the host command-template update.
    *(uint*)(uint64_t(pc.indirect) + uint64_t(pc.varyingByteOffset)) = n;
}
