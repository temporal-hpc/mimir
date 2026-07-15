#include "mimir/lod.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>  // std::min
#include <cstring>    // std::memcpy
#include <filesystem> // std::filesystem

#include "mimir/resources.hpp"
#include "mimir/shader.hpp"
#include "mimir/validation.hpp"

// Defined in pipeline.cpp (global namespace): resolves the directory holding the installed shaders
// so slang modules load by their "shaders/..." relative path.
std::string getDefaultShaderPath();

namespace mimir
{

namespace
{

constexpr VkMemoryPropertyFlags DEVICE_LOCAL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags HOST_VISIBLE =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

VkDeviceAddress getBufferAddress(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &info);
}

// Creates a buffer + memory. When want_address is set, the SHADER_DEVICE_ADDRESS usage bit and the
// matching allocate flag are added and the device address is resolved. Mirrors raytracing.cpp's
// makeBuffer (kept local so the LOD module is independent of the RT context).
RtBuffer makeBuffer(VkDevice device, VkPhysicalDeviceMemoryProperties mem_props, VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags, bool want_address)
{
    if (want_address) { usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; }

    RtBuffer buf{};
    buf.buffer = createBuffer(device, size, usage);

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buf.buffer, &req);

    VkMemoryAllocateFlagsInfo flags_info{
        .sType      = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext      = nullptr,
        .flags      = want_address ? VkMemoryAllocateFlags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) : 0u,
        .deviceMask = 0,
    };
    buf.memory = allocateMemory(device, mem_props, req, mem_flags,
        want_address ? &flags_info : nullptr);
    validation::checkVulkan(vkBindBufferMemory(device, buf.buffer, buf.memory, 0));

    if (want_address) { buf.address = getBufferAddress(device, buf.buffer); }
    return buf;
}

void destroyBuffer(VkDevice device, RtBuffer& buf)
{
    if (buf.buffer != VK_NULL_HANDLE) { vkDestroyBuffer(device, buf.buffer, nullptr); }
    if (buf.memory != VK_NULL_HANDLE) { vkFreeMemory(device, buf.memory, nullptr); }
    buf = {};
}

// Push constants for pathtrace_lod_scatter.slang: positions, per-cell count, and per-cell sum all as
// BDA pointers (offsets 0/8/16), then particle count, grid resolution, and the centroid flag. No
// descriptors: the count/sum accumulators are BDA (the sum exceeds maxStorageBufferRange at large N).
struct LodScatterPush
{
    VkDeviceAddress positions; VkDeviceAddress cellCounts; VkDeviceAddress cellSums;
    uint32_t count; uint32_t gridN; uint32_t centroid;
};
// Push constants for pathtrace_lod_emit.slang: reduced-position output, per-cell count, and per-cell
// sum as BDA pointers, then grid resolution and the centroid flag. Only the small global emit counter
// stays a descriptor (binding 0). The representative RADIUS is no longer here -- emit writes centroid
// POSITIONS; the radius is a consumer concern (the AABB writer / raster marker size).
struct LodEmitPush
{
    VkDeviceAddress reducedPos; VkDeviceAddress cellCounts; VkDeviceAddress cellSums;
    uint32_t gridN; uint32_t centroid;
};
// Push constants for lod_indirect_args.slang: the indirect-command buffer and the occupied-cell
// count as BDA pointers (offsets 0/8), then the byte offset of the command's varying field. The
// fixed field is pre-filled host-side (command template), so it is not passed here. Matches
// PushConstants in the shader.
struct LodIndirectPush
{
    VkDeviceAddress indirect; VkDeviceAddress count;
    uint32_t varyingByteOffset;
};

} // namespace

void LodContext::init(VkDevice dev, VkPhysicalDeviceMemoryProperties mp,
    bool int64_atomics, uint32_t grid, uint32_t particle_count)
{
    device    = dev;
    mem_props = mp;
    grid_n    = grid;

    const uint64_t num_cells = uint64_t(grid) * grid * grid;
    max_cells = static_cast<uint32_t>(std::min<uint64_t>(num_cells, particle_count));

    // Centroid placement needs int64 fixed-point atomics through a BDA pointer; when unavailable,
    // fall back to cell-center placement (no sum buffer).
    centroid_active = int64_atomics;

    // ---- Pipelines (scatter is descriptor-free; emit keeps a one-binding set for the counter) ----
    {
        // Emit descriptor set layout: a single STORAGE_BUFFER binding (the global emit counter).
        VkDescriptorSetLayoutBinding b{
            .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr };
        VkDescriptorSetLayoutCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .bindingCount = 1, .pBindings = &b };
        validation::checkVulkan(vkCreateDescriptorSetLayout(device, &info, nullptr, &emit_set_layout));
    }

    auto make_pipeline = [&](VkDescriptorSetLayout set_layout, uint32_t push_size,
                             const char* module, const char* entry,
                             VkPipelineLayout& out_layout, VkPipeline& out_pipe) {
        VkPushConstantRange range{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size };
        VkPipelineLayoutCreateInfo li{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .setLayoutCount = (set_layout != VK_NULL_HANDLE) ? 1u : 0u,
            .pSetLayouts = (set_layout != VK_NULL_HANDLE) ? &set_layout : nullptr,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
        validation::checkVulkan(vkCreatePipelineLayout(device, &li, nullptr, &out_layout));

        auto orig = std::filesystem::current_path();
        std::filesystem::current_path(getDefaultShaderPath());
        auto builder = ShaderBuilder::make();
        ShaderCompileParams params{ .module_path = module, .entrypoints = { entry }, .specializations = {} };
        auto stages = builder.compileModule(device, params);
        std::filesystem::current_path(orig);
        if (stages.size() != 1) { spdlog::error("{}: expected 1 compute stage, got {}", module, stages.size()); }

        VkComputePipelineCreateInfo pi{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .stage = stages[0], .layout = out_layout,
            .basePipelineHandle = VK_NULL_HANDLE, .basePipelineIndex = 0 };
        validation::checkVulkan(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &out_pipe));
        vkDestroyShaderModule(device, stages[0].module, nullptr);
    };
    make_pipeline(VK_NULL_HANDLE, sizeof(LodScatterPush),
        "shaders/pathtrace_lod_scatter.slang", "scatterMain", scatter_layout, scatter_pipeline);
    make_pipeline(emit_set_layout, sizeof(LodEmitPush),
        "shaders/pathtrace_lod_emit.slang", "emitMain", emit_layout, emit_pipeline);
    // Indirect-args finalize: descriptor-free (indirect + count buffers ride the push constants as BDA).
    make_pipeline(VK_NULL_HANDLE, sizeof(LodIndirectPush),
        "shaders/lod_indirect_args.slang", "finalizeMain", finalize_layout, finalize_pipeline);

    // One emit set per ringed slot (each binds that slot's HOST_VISIBLE emit counter). Scatter is
    // descriptor-free.
    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NUM_SLOTS };
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr, .flags = 0,
        .maxSets = NUM_SLOTS, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    validation::checkVulkan(vkCreateDescriptorPool(device, &pool_info, nullptr, &desc_pool));

    std::array<VkDescriptorSetLayout, NUM_SLOTS> set_layouts;
    set_layouts.fill(emit_set_layout);
    VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr,
        .descriptorPool = desc_pool, .descriptorSetCount = NUM_SLOTS, .pSetLayouts = set_layouts.data() };
    validation::checkVulkan(vkAllocateDescriptorSets(device, &ai, emit_set.data()));

    // ---- Buffers ----
    // Per-cell occupancy counts (one uint each), BDA (the sum below exceeds the descriptor cap at
    // large N, and the count moves to BDA alongside it), cleared each frame via vkCmdFillBuffer.
    cellcount_buffer = makeBuffer(device, mem_props, VkDeviceSize(num_cells) * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE_LOCAL, true);
    // Per-cell fixed-point position sum (3 * uint64 per cell) for centroid placement, BDA. Only
    // allocated when centroid is active; up to 3 * N^3 * 8 B (>4 GiB at large N), past the
    // maxStorageBufferRange descriptor cap -- hence BDA.
    if (centroid_active)
    {
        cellsum_buffer = makeBuffer(device, mem_props, VkDeviceSize(num_cells) * 3 * sizeof(uint64_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE_LOCAL, true);
    }

    // Per-slot RINGED output buffers (NUM_SLOTS copies each). These are the only buffers multi-buffered:
    //  - counter_buffer: emitted-primitive counter (host-readable).
    //  - reduced_pos: compacted list of occupied-cell centroids (float3), sized to the occupied bound.
    //    Usage covers every consumer: a raster vertex buffer, a BDA-read input to the path-tracer's AABB
    //    writer, a storage-buffer emit target, and a transfer destination.
    //  - indirect_buffer: a single Vk*IndirectCommand the raster draw sources via vkCmdDraw*Indirect.
    //    Sized to max(VkDrawIndirectCommand=16 B, VkDrawIndexedIndirectCommand=20 B) = 20 B so it fits
    //    either the point (non-indexed) or mesh (indexed) layout. recordIndirectArgs fills the fixed
    //    fields (TRANSFER_DST) then a compute pass writes the count (STORAGE); it is also an
    //    INDIRECT_BUFFER. BDA so the finalize shader addresses it as a raw pointer.
    for (uint32_t s = 0; s < NUM_SLOTS; ++s)
    {
        counter_buffer[s] = makeBuffer(device, mem_props, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_VISIBLE, true);
        reduced_pos[s] = makeBuffer(device, mem_props, VkDeviceSize(max_cells) * 3 * sizeof(float),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE_LOCAL, true);
        indirect_buffer[s] = makeBuffer(device, mem_props,
            std::max(sizeof(VkDrawIndirectCommand), sizeof(VkDrawIndexedIndirectCommand)),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE_LOCAL, true);

        // Point this slot's emit set global-counter binding (0) at this slot's counter buffer.
        VkDescriptorBufferInfo gc{ .buffer = counter_buffer[s].buffer, .offset = 0, .range = VK_WHOLE_SIZE };
        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = emit_set[s],
            .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
            .pBufferInfo = &gc, .pTexelBufferView = nullptr };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    spdlog::info("LOD: {}^3 grid, up to {} occupied cells (from {} particles), placement: {}",
        grid, max_cells, particle_count, centroid_active ? "centroid" : "cell-center (int64 atomics unavailable)");
}

void LodContext::recordReduction(VkCommandBuffer cmd, VkDeviceAddress positions_addr,
    uint32_t particle_count, uint32_t slot)
{
    // Record-only: the clear -> scatter -> emit passes go into `cmd`; the caller executes it (raster
    // inline in the frame cmd; PT in its own one-shot submit so it can readCount()). No internal
    // submit here, so raster incurs no host stall. Internal barriers (clear->scatter, scatter->emit)
    // stay here; the caller adds the trailing barrier for its own consumer. Outputs go to slot `slot`.
    const uint64_t num_cells = uint64_t(grid_n) * grid_n * grid_n;
    const uint32_t centroid_flag = centroid_active ? 1u : 0u;
    const VkDeviceAddress cellsum_addr = centroid_active ? cellsum_buffer.address : VkDeviceAddress(0);

    // Cross-frame accumulator serialization: cellcount_buffer/cellsum_buffer are SINGLE-buffered (the
    // N^3 memory hog stays single), so this frame's clear must not overwrite the accumulator while the
    // PREVIOUS frame's scatter/emit (a different slot, overlapping in flight) is still reading/writing
    // it -- a write-after-read/write hazard. Same queue + submission order means an execution + memory
    // dependency from prior COMPUTE (scatter/emit shader read+write) to this frame's TRANSFER (the clear
    // fill) orders us after the previous reduction's accumulator accesses. The ringed OUTPUT buffers
    // (reduced_pos/indirect/counter) are per-slot, so draws still overlap; only the reduction compute
    // is serialized. On the first frame there is no prior reduction, so this is a no-op. (PT wraps
    // recordReduction in its own vkQueueWaitIdle one-shot submit, so this barrier is redundant-but-
    // harmless there.)
    VkMemoryBarrier accum_serialize{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &accum_serialize, 0, nullptr, 0, nullptr);

    vkCmdFillBuffer(cmd, cellcount_buffer.buffer,     0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, counter_buffer[slot].buffer, 0, VK_WHOLE_SIZE, 0u);
    if (centroid_active) { vkCmdFillBuffer(cmd, cellsum_buffer.buffer, 0, VK_WHOLE_SIZE, 0u); }
    VkMemoryBarrier clr{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &clr, 0, nullptr, 0, nullptr);

    LodScatterPush sp{ .positions = positions_addr, .cellCounts = cellcount_buffer.address,
        .cellSums = cellsum_addr, .count = particle_count, .gridN = grid_n, .centroid = centroid_flag };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scatter_pipeline);
    vkCmdPushConstants(cmd, scatter_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
    vkCmdDispatch(cmd, (particle_count + 63) / 64, 1, 1);

    VkMemoryBarrier s2e{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &s2e, 0, nullptr, 0, nullptr);

    LodEmitPush ep{ .reducedPos = reduced_pos[slot].address, .cellCounts = cellcount_buffer.address,
        .cellSums = cellsum_addr, .gridN = grid_n, .centroid = centroid_flag };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, emit_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, emit_layout, 0, 1, &emit_set[slot], 0, nullptr);
    vkCmdPushConstants(cmd, emit_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ep), &ep);
    vkCmdDispatch(cmd, static_cast<uint32_t>((num_cells + 63) / 64), 1, 1);
}

void LodContext::recordIndirectArgs(VkCommandBuffer cmd, uint32_t fixed_byte_offset,
    uint32_t fixed_value, uint32_t varying_byte_offset, uint32_t slot)
{
    // 1) Write the WHOLE command template in one transfer: all fields zero (firstVertex/firstIndex/
    // vertexOffset/firstInstance = 0, and the varying field a placeholder 0 the finalize dispatch
    // overwrites) except the FIXED field, which gets `fixed_value` at `fixed_byte_offset` (point:
    // instanceCount@4=1; mesh: indexCount@0=icosphere index count). Five uints (20 B) cover both the
    // 16 B non-indexed and 20 B indexed layouts. vkCmdUpdateBuffer is a single TRANSFER write, so no
    // fill/update intra-transfer ordering hazard. Reads/writes slot `slot`'s ringed buffers.
    uint32_t cmd_template[5] = { 0u, 0u, 0u, 0u, 0u };
    cmd_template[fixed_byte_offset / sizeof(uint32_t)] = fixed_value;
    vkCmdUpdateBuffer(cmd, indirect_buffer[slot].buffer, 0, sizeof(cmd_template), cmd_template);
    VkMemoryBarrier fill_to_finalize{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &fill_to_finalize, 0, nullptr, 0, nullptr);

    // 2) One thread copies the occupied count into the varying field. Reads the same HOST_VISIBLE emit
    // counter the emit pass wrote -- the caller must have made that write visible (the emit's compute
    // write precedes this dispatch on the same queue; recordReduction's own emit runs earlier in `cmd`,
    // and the raster call site adds the emit->args barrier before invoking this).
    LodIndirectPush ip{ .indirect = indirect_buffer[slot].address, .count = counter_buffer[slot].address,
        .varyingByteOffset = varying_byte_offset };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, finalize_pipeline);
    vkCmdPushConstants(cmd, finalize_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ip), &ip);
    vkCmdDispatch(cmd, 1, 1, 1);

    // 3) Make the finished command visible to the indirect-draw consumer.
    VkMemoryBarrier args_to_draw{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0, 1, &args_to_draw, 0, nullptr, 0, nullptr);
}

uint32_t LodContext::readCount(uint32_t slot)
{
    uint32_t occupied = 0;
    void* mapped = nullptr;
    validation::checkVulkan(vkMapMemory(device, counter_buffer[slot].memory, 0, sizeof(uint32_t), 0, &mapped));
    std::memcpy(&occupied, mapped, sizeof(uint32_t));
    vkUnmapMemory(device, counter_buffer[slot].memory);
    return occupied;
}

float LodContext::sphereRadius(float default_size) const
{
    // cellFill reproduces the old cell-derived radius (LOD_COVERAGE * cellSize / 2) at the reference
    // --size; scaling by default_size / LOD_REFERENCE_SIZE makes --size a live multiplier.
    const float cell_fill = LOD_COVERAGE * (2.0f / float(grid_n)) * 0.5f;
    return cell_fill * (default_size / LOD_REFERENCE_SIZE);
}

void LodContext::destroy()
{
    if (scatter_pipeline   != VK_NULL_HANDLE) { vkDestroyPipeline(device, scatter_pipeline, nullptr); }
    if (emit_pipeline      != VK_NULL_HANDLE) { vkDestroyPipeline(device, emit_pipeline, nullptr); }
    if (finalize_pipeline  != VK_NULL_HANDLE) { vkDestroyPipeline(device, finalize_pipeline, nullptr); }
    if (scatter_layout     != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, scatter_layout, nullptr); }
    if (emit_layout        != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, emit_layout, nullptr); }
    if (finalize_layout    != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, finalize_layout, nullptr); }
    if (emit_set_layout    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, emit_set_layout, nullptr); }
    if (desc_pool          != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device, desc_pool, nullptr); }
    scatter_pipeline = emit_pipeline = finalize_pipeline = VK_NULL_HANDLE;
    scatter_layout = emit_layout = finalize_layout = VK_NULL_HANDLE;
    emit_set_layout = VK_NULL_HANDLE;
    desc_pool = VK_NULL_HANDLE;
    emit_set.fill(VK_NULL_HANDLE);

    destroyBuffer(device, cellcount_buffer);
    destroyBuffer(device, cellsum_buffer);
    for (uint32_t s = 0; s < NUM_SLOTS; ++s)
    {
        destroyBuffer(device, counter_buffer[s]);
        destroyBuffer(device, reduced_pos[s]);
        destroyBuffer(device, indirect_buffer[s]);
    }
    grid_n = 0;
}

} // namespace mimir
