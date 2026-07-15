#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional> // std::function

#include "mimir/raytracing.hpp" // RtBuffer (shared device-buffer helper type)

namespace mimir
{

// Shared LOD (level-of-detail) data-reduction stage. Bins the live particle positions into an
// N^3 grid over the fixed [-1,1]^3 domain (scatter), then compacts one representative POSITION per
// occupied cell into a reduced-position buffer (emit). The representative is the per-cell mass
// centroid when int64 BDA atomics are available, else the cell's geometric centre.
//
// This is a pre-render stage independent of the render mode: it produces a reduced particle set
// (positions + count) that any consumer draws. Path-tracing feeds the reduced positions to its AABB
// writer; raster point modes (none/phong) bind the reduced buffer as a vertex buffer and draw it via
// vkCmdDrawIndirect, sourcing the count from the GPU indirect-args buffer (recordIndirectArgs).
//
// Extracted from raytracing.cpp's former in-pipeline scatter/emit (the count/centroids are
// unchanged by the move -- same integer atomics, same reduction). See
// docs/superpowers/specs/2026-07-15-lod-transversal-render-modes-design.md.
class LodContext
{
public:
    using SubmitFn = std::function<void(std::function<void(VkCommandBuffer)>)>;

    // Reference world radius (--size) at which the LOD sphere exactly fills the cell (cellFill),
    // matching the pre-transversal opaque look. It is the LIT-mode DEFAULT --size world value from
    // samples/remote-rendering/rr-server.cu: the default --size is size_px=5, and a lit view maps
    // size = size_px/100 = 0.05 world units (rr-server.cu: `size = size_px / 100.f`). So at the
    // default --size, sphereRadius() == cellFill (== the old cell-derived radius) and the image is
    // unchanged; larger/smaller --size scales the blobs proportionally (--size becomes live).
    static constexpr float LOD_REFERENCE_SIZE = 0.05f;

    // Sphere radius as a fraction of the cell that a representative fills. The old (dead --size) LOD
    // radius was exactly LOD_COVERAGE * cellSize / 2 = LOD_COVERAGE * (2/N) * 0.5; sphereRadius()
    // reproduces that at the reference --size.
    static constexpr float LOD_COVERAGE = 1.2f;

    // Fixed-point scale for the centroid position sum: maps [-1,1] -> [0, 2^30]. Integer atomics are
    // order-independent (deterministic); 2^30 keeps a sum of ~5*10^8 particles inside int64. Must
    // match SCALE in pathtrace_lod_scatter.slang / pathtrace_lod_emit.slang.
    static constexpr double LOD_FIXEDPOINT_SCALE = 1073741824.0; // 2^30

    // Build the LOD stage: create the scatter/emit compute pipelines and allocate the accumulator,
    // counter, and reduced-position buffers for an N^3 grid over `particle_count` particles.
    // `int64_atomics` gates centroid placement (else cell-center fallback). Idempotent per instance;
    // call once before the first recordReduction. active() is true afterwards (grid_n = N > 0).
    void init(VkDevice device, VkPhysicalDeviceMemoryProperties mem_props, SubmitFn submit,
        bool int64_atomics, uint32_t grid_n, uint32_t particle_count);

    // Record the reduction for this frame INTO `cmd` (clear -> scatter -> emit), reading the live
    // positions at `positions_addr` (tightly-packed float3, BDA). Writes the reduced positions + the
    // occupied-cell counter. This is now RECORD-ONLY (no internal submit): the caller decides how to
    // execute it. Raster records it inline in the frame command buffer (no host stall); path tracing
    // wraps it in its own one-shot submit so it can readCount() before building the AS. The caller is
    // responsible for the barrier that makes the emit's reduced-position writes visible to its
    // consumer (vertex-input read for raster, AABB-writer read for PT).
    void recordReduction(VkCommandBuffer cmd, VkDeviceAddress positions_addr, uint32_t particle_count);

    // Record the indirect-args build INTO `cmd`: fill the command template (firstVertex/firstInstance
    // = 0), then a 1-thread compute dispatch writes the occupied count into the command's varying
    // field at `varying_byte_offset` and `fixed_instance_count` into the instanceCount field. Ends
    // with a barrier (SHADER_WRITE -> INDIRECT_COMMAND_READ, dstStage DRAW_INDIRECT) so a following
    // vkCmdDraw*Indirect from indirectBuffer() reads the finished command. No host readback.
    void recordIndirectArgs(VkCommandBuffer cmd, uint32_t fixed_instance_count,
        uint32_t varying_byte_offset);

    // Read back the emitted occupied-cell count (HOST_VISIBLE, coherent). Call after recordReduction
    // has EXECUTED (path tracing's one-shot submit). NOT clamped to maxCells(); the consumer clamps.
    uint32_t readCount();

    // The compacted occupied-cell representative positions (float3[], BDA + vertex buffer).
    VkBuffer        reducedPositionsBuffer()  const { return reduced_pos.buffer; }
    VkDeviceAddress reducedPositionsAddress() const { return reduced_pos.address; }

    // The GPU-resident VkDrawIndirectCommand written by recordIndirectArgs (INDIRECT_BUFFER usage).
    VkBuffer indirectBuffer() const { return indirect_buffer.buffer; }

    // World radius of an LOD representative sphere given the view's default --size (lit world value).
    // = cellFill * (default_size / LOD_REFERENCE_SIZE), cellFill = LOD_COVERAGE * (2/N) * 0.5.
    float sphereRadius(float default_size) const;

    bool     active()   const { return grid_n > 0; }
    uint32_t cells()    const { return grid_n; }
    // min(N^3, particle_count): the sizing bound for the reduced set (and the consumer's AABB/BLAS).
    uint32_t maxCells() const { return max_cells; }
    bool     centroid() const { return centroid_active; }

    // Teardown (safe to call when uninitialized: no-ops on null handles).
    void destroy();

private:
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    SubmitFn submit;

    uint32_t grid_n = 0;          // cells per axis (N); 0 = inactive
    uint32_t max_cells = 0;       // min(N^3, particle_count)
    bool     centroid_active = false; // centroid placement (int64 atomics available) vs cell-center

    // Accumulators (BDA): per-cell occupancy count and the fixed-point position sum (centroid only),
    // plus the small HOST_VISIBLE emit counter. Reduced positions: compacted representative float3[].
    RtBuffer cellcount_buffer; // N^3 uint occupancy counts (DEVICE_LOCAL, BDA)
    RtBuffer cellsum_buffer;   // 3 * N^3 uint64 fixed-point sums (DEVICE_LOCAL, BDA; centroid only)
    RtBuffer counter_buffer;   // 1 uint emitted-primitive counter (HOST_VISIBLE, readback)
    RtBuffer reduced_pos;      // min(N^3,P) float3 representative positions (DEVICE_LOCAL, BDA + VBO)
    RtBuffer indirect_buffer;  // 1 VkDrawIndirectCommand (16 B) for raster indirect draw (DEVICE_LOCAL, BDA)

    // Scatter/emit compute pipelines. Scatter binds no descriptors (all accumulators are BDA); emit
    // keeps one descriptor for the global emit counter (binding 0). Finalize (indirect-args build) is
    // descriptor-free (indirect + count are BDA push-constant pointers).
    VkDescriptorSetLayout emit_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout      scatter_layout  = VK_NULL_HANDLE;
    VkPipelineLayout      emit_layout     = VK_NULL_HANDLE;
    VkPipelineLayout      finalize_layout = VK_NULL_HANDLE;
    VkPipeline            scatter_pipeline  = VK_NULL_HANDLE;
    VkPipeline            emit_pipeline     = VK_NULL_HANDLE;
    VkPipeline            finalize_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       emit_set        = VK_NULL_HANDLE;
};

} // namespace mimir
