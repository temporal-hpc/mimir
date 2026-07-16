#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>

#include "mimir/lod_reduce.hpp"  // LodReduce (native-CUDA reduction, pure C++ pImpl facade)
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
    // Number of frame-in-flight slots the per-frame OUTPUT buffers are ringed over. Must equal
    // MimirInstance::MAX_FRAMES_IN_FLIGHT (engine.hpp): the engine passes render_timeline %
    // MAX_FRAMES_IN_FLIGHT as the slot so frame T's draw reads the SAME slot frame T's reduction wrote.
    // Only the small particle-bounded outputs (reduced_pos, indirect, counter) are ringed; the N^3
    // accumulator (cellcount/cellsum -- the memory hog) stays single and is cross-frame serialized.
    static constexpr uint32_t NUM_SLOTS = 3;

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
    // Centroid placement is used only when BOTH the hardware has int64 atomics (`int64_atomics`)
    // AND the caller requests it (`want_centroid`); otherwise it falls back to cell-center placement
    // (no sum buffer, no int64 atomics in the scatter). Call-once: there is no idempotency guard, so
    // a second call would leak the pipelines/buffers allocated by the first (callers already
    // guarantee init() runs at most once per instance). active() is true afterwards (grid_n = N > 0).
    void init(VkDevice device, VkPhysicalDeviceMemoryProperties mem_props,
        bool int64_atomics, bool want_centroid, uint32_t grid_n, uint64_t particle_count);

    // Record the reduction for this frame INTO `cmd` (clear -> scatter -> emit), reading the live
    // positions at `positions_addr` (tightly-packed float3, BDA). Writes the reduced positions + the
    // occupied-cell counter. This is now RECORD-ONLY (no internal submit): the caller decides how to
    // execute it. Raster records it inline in the frame command buffer (no host stall); path tracing
    // wraps it in its own one-shot submit so it can readCount() before building the AS. The caller is
    // responsible for the barrier that makes the emit's reduced-position writes visible to its
    // consumer (vertex-input read for raster, AABB-writer read for PT).
    //
    // `slot` (0..NUM_SLOTS-1) selects which ringed OUTPUT buffers (reduced_pos, counter) receive this
    // frame's reduction; the engine passes render_timeline % MAX_FRAMES_IN_FLIGHT so overlapping frames
    // never clobber each other's outputs. PT serializes itself and uses a fixed slot 0. The N^3
    // accumulator is shared across slots, so this records a cross-frame serialize barrier at the START
    // (prior-frame COMPUTE read/write on the accumulator -> this frame's TRANSFER clear) before the
    // clear fill, ordering this reduction after the previous frame's scatter/emit still reading it.
    void recordReduction(VkCommandBuffer cmd, VkDeviceAddress positions_addr, uint64_t particle_count,
        uint32_t slot);

    // Record the indirect-args build INTO `cmd` for EITHER command layout. The host writes the whole
    // command template (vkCmdUpdateBuffer): all bytes zero except `fixed_value` placed at
    // `fixed_byte_offset` (its FIXED field). Then a 1-thread compute dispatch writes the occupied
    // count into the command's VARYING field at `varying_byte_offset`. This generalizes both modes:
    //   - point (VkDrawIndirectCommand):        fixed = instanceCount@4 = 1, varying = vertexCount@0.
    //   - mesh  (VkDrawIndexedIndirectCommand):  fixed = indexCount@0    = N, varying = instanceCount@4.
    // Ends with a barrier (SHADER_WRITE -> INDIRECT_COMMAND_READ, dstStage DRAW_INDIRECT) so a
    // following vkCmdDraw*Indirect from indirectBuffer() reads the finished command. No host readback.
    void recordIndirectArgs(VkCommandBuffer cmd, uint32_t fixed_byte_offset, uint32_t fixed_value,
        uint32_t varying_byte_offset, uint32_t slot);

    // Read back slot `slot`'s emitted occupied-cell count (HOST_VISIBLE, coherent). Call after
    // recordReduction on that slot has EXECUTED (path tracing's one-shot submit). NOT clamped to
    // maxCells(); the consumer clamps.
    uint32_t readCount(uint32_t slot);

    // The compacted occupied-cell representative positions for `slot` (float3[], BDA + vertex buffer).
    VkBuffer        reducedPositionsBuffer(uint32_t slot)  const { return reduced_pos[slot].buffer; }
    VkDeviceAddress reducedPositionsAddress(uint32_t slot) const { return reduced_pos[slot].address; }

    // The GPU-resident VkDrawIndirectCommand written by recordIndirectArgs for `slot` (INDIRECT_BUFFER).
    VkBuffer indirectBuffer(uint32_t slot) const { return indirect_buffer[slot].buffer; }

    // CUDA device pointers aliasing this slot's reduced-position / occupied-count buffers, imported
    // via external memory when the CUDA reduction path is active (`use_cuda`); nullptr otherwise (the
    // default Vulkan-only path, where LodReduce is never invoked and these aliases are never created).
    void*     reducedPositionsDevicePtr(uint32_t slot) const { return reduced_pos_cuda[slot]; }
    uint32_t* occupiedDevicePtr(uint32_t slot)          const { return occupied_cuda[slot]; }

    // True when the CUDA reduction path is active (the default; false only under
    // MIMIR_LOD_NO_CUDA, which forces the Vulkan scatter/emit fallback).
    bool usesCuda() const { return use_cuda; }

    // Frame/sim coupling for the CUDA reduction. When decoupled (the remote server's sovereign-sim
    // mode), the reduction runs on a DEDICATED stream so its blocking syncReduce() waits only for the
    // reduction, never for the sim's stream -- preserving "the viewer never slows the run" and the
    // torn-latest read contract. When coupled (lockstep / windowed display, the default), the
    // reduction runs on the caller-provided sim stream so it is naturally ordered AFTER the sim's
    // writes (tear-free). Default false (coupled) is the safe choice for every non-decoupled path.
    void setDecoupledReduction(bool decoupled) { lod_decoupled = decoupled; }

    // Run the CUDA reduction for `slot`: reads `count` positions from `positions_dev` (device ptr,
    // packed float3), writes this slot's reduced-position + occupied-count CUDA aliases. Runs on the
    // dedicated reduce stream when decoupled (see setDecoupledReduction), else on `sim_stream` (so it
    // is ordered after the sim). Pair every call with syncReduce() before reading the outputs. No-op
    // when !usesCuda() -- must not be called on the Vulkan-fallback path.
    void reduceCuda(cudaStream_t sim_stream, const void* positions_dev, uint64_t count, uint32_t slot);

    // Block the host until the most recent reduceCuda() (on whichever stream it chose) has completed,
    // so its reduced-position/occupied-count writes are final before the Vulkan renderer reads them
    // and before occupiedFromCuda(). No-op when !usesCuda().
    void syncReduce();

    // Device->host copy of slot `slot`'s CUDA-reduced occupied-cell count, clamped to maxCells().
    // The caller must have already called syncReduce() (or otherwise ordered against the reduction)
    // before calling this -- the copy itself is a plain blocking cudaMemcpy.
    uint32_t occupiedFromCuda(uint32_t slot);

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

    uint32_t grid_n = 0;          // cells per axis (N); 0 = inactive
    uint32_t max_cells = 0;       // min(N^3, particle_count)
    bool     centroid_active = false; // centroid placement (int64 atomics available) vs cell-center

    // Selects the CUDA-primary reduction path: true unless MIMIR_LOD_NO_CUDA is set (see init()). When
    // true, init() constructs `lod_reduce`, imports the CUDA device-pointer aliases for
    // reduced_pos[]/counter_buffer[], and SKIPS allocating the Vulkan N^3 accumulator
    // (cellcount_buffer/cellsum_buffer) -- LodReduce owns the one and only N^3 accumulator. When false
    // (the MIMIR_LOD_NO_CUDA fallback), the Vulkan scatter/emit path and its N^3 accumulator are used
    // exactly as before, unchanged.
    bool use_cuda = false;

    // Native-CUDA reduction (Task 1). Owns its own N^3 accumulator; constructed in init() only when
    // use_cuda is true. Non-copyable, so held by pointer; reset() in destroy() to free its accumulator.
    std::unique_ptr<LodReduce> lod_reduce;

    // Dedicated CUDA stream the reduction runs on in DECOUPLED mode, so the render thread's blocking
    // syncReduce() waits only for the reduction and not for the sovereign sim's (default-stream) work.
    // Created in init() when use_cuda, destroyed in destroy(). In coupled (lockstep/default) mode the
    // reduction instead runs on the caller's sim stream, so this stream stays idle.
    cudaStream_t reduce_stream = nullptr;
    // Whether the reduction is decoupled from the sim (torn-latest reads, independent). Default false
    // (coupled: ordered after the sim on its stream => tear-free). Set by setDecoupledReduction().
    bool lod_decoupled = false;
    // The stream the most recent reduceCuda() actually ran on; syncReduce() blocks on this one.
    cudaStream_t active_reduce_stream = nullptr;

    // Accumulators (BDA): per-cell occupancy count and the fixed-point position sum (centroid only).
    // SINGLE-buffered (shared across frame slots): this is the N^3 memory hog (~30 GB at 1024^3) and is
    // per-frame scratch (cleared -> scattered -> emitted, never read cross-frame), so it is not ringed;
    // recordReduction serializes the reduction across frames instead (a cross-frame barrier on it).
    RtBuffer cellcount_buffer; // N^3 uint occupancy counts (DEVICE_LOCAL, BDA)
    RtBuffer cellsum_buffer;   // 3 * N^3 uint64 fixed-point sums (DEVICE_LOCAL, BDA; centroid only)

    // Per-frame OUTPUT buffers -- RINGED over NUM_SLOTS so frame T's draw reads the same slot frame T's
    // reduction wrote while frame T+1's reduction targets a different slot (no cross-frame WAR). These
    // are particle-bounded (small), so tripling them is cheap.
    std::array<RtBuffer, NUM_SLOTS> counter_buffer;   // 1 uint emitted-primitive counter (HOST_VISIBLE, readback)
    std::array<RtBuffer, NUM_SLOTS> reduced_pos;      // min(N^3,P) float3 representative positions (DEVICE_LOCAL, BDA + VBO)
    std::array<RtBuffer, NUM_SLOTS> indirect_buffer;  // 1 Vk*IndirectCommand (max(16,20)=20 B) for raster indirect draw (DEVICE_LOCAL, BDA)

    // CUDA external-memory handles + mapped device-pointer aliases for counter_buffer[]/reduced_pos[],
    // populated only when use_cuda is true (see init()). Torn down in destroy() via
    // cudaDestroyExternalMemory before the aliased Vulkan memory is freed; the mapped pointers
    // themselves are never cudaFree'd (release is via destroying the external memory, same as the
    // engine's interop position buffer -- see engine.cpp's allocLinear).
    std::array<cudaExternalMemory_t, NUM_SLOTS> reduced_pos_extmem{};
    std::array<cudaExternalMemory_t, NUM_SLOTS> counter_extmem{};
    std::array<void*, NUM_SLOTS>     reduced_pos_cuda{}; // CUDA ptr aliasing reduced_pos[slot]
    std::array<uint32_t*, NUM_SLOTS> occupied_cuda{};    // CUDA ptr aliasing counter_buffer[slot]

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
    // One emit descriptor set per slot: each binds its slot's (ringed) HOST_VISIBLE emit counter.
    std::array<VkDescriptorSet, NUM_SLOTS> emit_set{};
};

} // namespace mimir
