#include "mimir/raytracing.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>     // std::max
#include <cassert>       // assert (LOD single-chunk invariant)
#include <cstring>       // std::memcpy
#include <cstdlib>       // std::getenv, std::strtoull (MIMIR_PT_BLAS_CHUNK)
#include <cmath>         // std::sqrt
#include <filesystem>    // std::filesystem
#include <unordered_map> // std::unordered_map

#include "mimir/resources.hpp"
#include "mimir/shader.hpp"
#include "mimir/validation.hpp"

// Defined in pipeline.cpp (global namespace): resolves the directory holding the
// installed shaders so slang modules load by their "shaders/..." relative path.
std::string getDefaultShaderPath();

namespace mimir
{

RayTracingApi RayTracingApi::load(VkDevice device)
{
    RayTracingApi api{};
    #define LOAD(field, name) \
        api.field = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name)); \
        if (api.field == nullptr) { spdlog::error("Failed to load {}", #name); }
    LOAD(createAccelerationStructure,   vkCreateAccelerationStructureKHR);
    LOAD(destroyAccelerationStructure,  vkDestroyAccelerationStructureKHR);
    LOAD(getBuildSizes,                 vkGetAccelerationStructureBuildSizesKHR);
    LOAD(cmdBuildAccelerationStructures,vkCmdBuildAccelerationStructuresKHR);
    LOAD(getAccelStructAddress,         vkGetAccelerationStructureDeviceAddressKHR);
    LOAD(createRayTracingPipelines,     vkCreateRayTracingPipelinesKHR);
    LOAD(getShaderGroupHandles,         vkGetRayTracingShaderGroupHandlesKHR);
    LOAD(cmdTraceRays,                  vkCmdTraceRaysKHR);
    #undef LOAD
    return api;
}

namespace
{

constexpr VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

VkDeviceAddress getBufferAddress(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext  = nullptr,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &info);
}

// Creates a buffer + memory. When want_address is set, the SHADER_DEVICE_ADDRESS usage
// bit and the matching allocate flag are added and the device address is resolved.
RtBuffer makeBuffer(RayTracingContext& ctx, VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags, bool want_address)
{
    if (want_address) { usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; }

    RtBuffer buf{};
    buf.buffer = createBuffer(ctx.device, size, usage);

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &req);

    VkMemoryAllocateFlagsInfo flags_info{
        .sType      = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext      = nullptr,
        .flags      = want_address ? VkMemoryAllocateFlags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT) : 0u,
        .deviceMask = 0,
    };
    buf.memory = allocateMemory(ctx.device, ctx.mem_props, req, mem_flags,
        want_address ? &flags_info : nullptr);
    validation::checkVulkan(vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0));

    if (want_address) { buf.address = getBufferAddress(ctx.device, buf.buffer); }
    return buf;
}

void uploadBuffer(VkDevice device, const RtBuffer& buf, const void* data, VkDeviceSize size)
{
    void* mapped = nullptr;
    validation::checkVulkan(vkMapMemory(device, buf.memory, 0, size, 0, &mapped));
    std::memcpy(mapped, data, size);
    vkUnmapMemory(device, buf.memory);
}

void destroyBuffer(VkDevice device, RtBuffer& buf)
{
    if (buf.buffer != VK_NULL_HANDLE) { vkDestroyBuffer(device, buf.buffer, nullptr); }
    if (buf.memory != VK_NULL_HANDLE) { vkFreeMemory(device, buf.memory, nullptr); }
    buf = {};
}

constexpr VkMemoryPropertyFlags DEVICE_LOCAL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags HOST_VISIBLE =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;


// ---- Dynamic (per-frame) BLAS of procedural AABB spheres --------------------------

// Builds the BLAS geometry/build-info for `count` procedural AABBs at `aabb_addr` (stride 24 B =
// sizeof(VkAabbPositionsKHR)). The returned geometry must stay alive for the build call.
VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo(
    VkAccelerationStructureGeometryKHR& geometry, VkDeviceAddress aabb_addr)
{
    geometry = VkAccelerationStructureGeometryKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
        .geometry = {},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    geometry.geometry.aabbs = VkAccelerationStructureGeometryAabbsDataKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
        .pNext = nullptr,
        .data = { .deviceAddress = aabb_addr },
        .stride = sizeof(VkAabbPositionsKHR),
    };
    return VkAccelerationStructureBuildGeometryInfoKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        // PREFER_FAST_BUILD: the full BLAS rebuild dominates the frame at large N, so build speed
        // matters more than a marginally faster trace. ALLOW_UPDATE: permits cheap in-place refits
        // (VK_..._MODE_UPDATE) on the frames between full rebuilds -- see recordUpdateScene. The flags
        // must match on the build that sized the AS (createDynamicBlasChunks) and every (re)build/refit.
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
               | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
        .mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries   = &geometry,
        .ppGeometries  = nullptr,
        .scratchData   = {},
    };
}

// Device address of AABB chunk `c` within the shared buffer (chunk = blas_chunk_prims AABBs).
VkDeviceAddress chunkAabbAddr(const RayTracingContext& ctx, VkDeviceAddress aabb_addr, uint32_t c)
{
    return aabb_addr + VkDeviceSize(c) * ctx.blas_chunk_prims * sizeof(VkAabbPositionsKHR);
}

// Primitive count of chunk `c` (all full-sized except possibly the last).
uint32_t chunkPrimCount(const RayTracingContext& ctx, uint32_t c)
{
    uint32_t base = c * ctx.blas_chunk_prims;
    uint32_t rem = ctx.particle_count - base;
    return rem < ctx.blas_chunk_prims ? rem : ctx.blas_chunk_prims;
}

// Allocates ceil(count / blas_chunk_prims) BLASes over consecutive AABB chunks, plus ONE shared build
// scratch sized for the largest chunk. maxPrimitiveCount is a per-BLAS limit; chunking lifts the
// overall particle cap to (num_chunks * maxPrimitiveCount), i.e. VRAM. Does not build yet.
void createDynamicBlasChunks(RayTracingContext& ctx, VkDeviceAddress aabb_addr, uint32_t count)
{
    uint32_t num_chunks = (count + ctx.blas_chunk_prims - 1) / ctx.blas_chunk_prims;
    ctx.scene_blas.assign(num_chunks, AccelStruct{});
    VkDeviceSize max_scratch = 0;

    for (uint32_t c = 0; c < num_chunks; ++c)
    {
        uint32_t prims = chunkPrimCount(ctx, c);
        VkAccelerationStructureGeometryKHR geometry{};
        auto build_info = blasBuildInfo(geometry, chunkAabbAddr(ctx, aabb_addr, c));
        VkAccelerationStructureBuildSizesInfoKHR sizes{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
            .pNext = nullptr, .accelerationStructureSize = 0,
            .updateScratchSize = 0, .buildScratchSize = 0,
        };
        ctx.api.getBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_info, &prims, &sizes);

        AccelStruct& blas = ctx.scene_blas[c];
        blas.buffer = makeBuffer(ctx, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, DEVICE_LOCAL, true);
        VkAccelerationStructureCreateInfoKHR create_info{
            .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .pNext  = nullptr, .createFlags = 0, .buffer = blas.buffer.buffer, .offset = 0,
            .size   = sizes.accelerationStructureSize,
            .type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, .deviceAddress = 0,
        };
        validation::checkVulkan(ctx.api.createAccelerationStructure(
            ctx.device, &create_info, nullptr, &blas.handle));
        VkAccelerationStructureDeviceAddressInfoKHR addr_info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .pNext = nullptr, .accelerationStructure = blas.handle,
        };
        blas.address = ctx.api.getAccelStructAddress(ctx.device, &addr_info);

        // Size the shared scratch for the larger of a full build and an in-place refit (UPDATE). In
        // practice updateScratchSize <= buildScratchSize, but take the max so a refit can never
        // overrun the scratch on any driver.
        max_scratch = std::max({ max_scratch, sizes.buildScratchSize, sizes.updateScratchSize });
    }

    auto scratch_align = ctx.accel_props.minAccelerationStructureScratchOffsetAlignment;
    ctx.blas_scratch = makeBuffer(ctx, max_scratch + scratch_align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, DEVICE_LOCAL, true);
}

// Records the (re)build of every chunk BLAS from its AABB slice. The chunks share one scratch buffer,
// so a barrier serializes each build after the previous one (write-after-write/read on the scratch).
// `update` = refit the existing BVH in place (MODE_UPDATE, src == dst) instead of a full rebuild:
// far cheaper, but keeps the topology built by the last full rebuild (the caller bounds how stale
// that gets). The shared scratch is sized for the largest full build, so it also covers any update
// (updateScratchSize <= buildScratchSize).
void recordBlasBuildChunks(RayTracingContext& ctx, VkCommandBuffer cmd, VkDeviceAddress aabb_addr,
    bool update, uint32_t override_prims = 0)
{
    auto scratch_align = ctx.accel_props.minAccelerationStructureScratchOffsetAlignment;
    VkDeviceAddress scratch_addr = alignUp(ctx.blas_scratch.address, scratch_align);
    uint32_t num_chunks = static_cast<uint32_t>(ctx.scene_blas.size());

    for (uint32_t c = 0; c < num_chunks; ++c)
    {
        // LOD builds a single chunk over the compacted occupied-cell list, whose length varies per
        // frame; override_prims (> 0) supplies that count. The per-particle path passes 0 and uses
        // the fixed chunk size.
        uint32_t prims = (override_prims > 0) ? override_prims : chunkPrimCount(ctx, c);
        VkAccelerationStructureGeometryKHR geometry{};
        auto build_info = blasBuildInfo(geometry, chunkAabbAddr(ctx, aabb_addr, c));
        build_info.dstAccelerationStructure = ctx.scene_blas[c].handle;
        if (update)
        {
            // In-place refit: source and destination are the same AS (allowed with ALLOW_UPDATE).
            build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            build_info.srcAccelerationStructure = ctx.scene_blas[c].handle;
        }
        build_info.scratchData.deviceAddress = scratch_addr;

        VkAccelerationStructureBuildRangeInfoKHR range{
            .primitiveCount = prims, .primitiveOffset = 0, .firstVertex = 0, .transformOffset = 0,
        };
        const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
        ctx.api.cmdBuildAccelerationStructures(cmd, 1, &build_info, &p_range);

        // Serialize the next chunk's build on the shared scratch (and this chunk's AS is complete).
        if (c + 1 < num_chunks)
        {
            VkMemoryBarrier b{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                               | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &b, 0, nullptr, 0, nullptr);
        }
    }
}

// ---- Dynamic (per-frame) TLAS from a live instance buffer ------------------------

// Builds the TLAS geometry/build-info for `count` instances at `instances_addr`. The
// returned geometry must stay alive for the build call (it is referenced by build_info).
VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo(
    VkAccelerationStructureGeometryKHR& geometry, VkDeviceAddress instances_addr)
{
    geometry = VkAccelerationStructureGeometryKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = {},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    geometry.geometry.instances = VkAccelerationStructureGeometryInstancesDataKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .pNext = nullptr,
        .arrayOfPointers = VK_FALSE,
        .data = { .deviceAddress = instances_addr },
    };
    return VkAccelerationStructureBuildGeometryInfoKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        // PREFER_FAST_BUILD: the TLAS is rebuilt from scratch every frame, so build speed
        // matters more than a marginally faster trace (design §4).
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
        .mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries   = &geometry,
        .ppGeometries  = nullptr,
        .scratchData   = {},
    };
}

// Allocates a per-frame TLAS (AS backing + persistent build scratch) sized for `count`
// instances. Does not build it yet; the first build happens via recordTlasBuild.
void createDynamicTlas(RayTracingContext& ctx, AccelStruct& tlas, RtBuffer& scratch,
    VkDeviceAddress instances_addr, uint32_t count)
{
    VkAccelerationStructureGeometryKHR geometry{};
    auto build_info = tlasBuildInfo(geometry, instances_addr);

    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
        .pNext = nullptr, .accelerationStructureSize = 0,
        .updateScratchSize = 0, .buildScratchSize = 0,
    };
    ctx.api.getBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &count, &sizes);

    tlas.buffer = makeBuffer(ctx, sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, DEVICE_LOCAL, true);

    VkAccelerationStructureCreateInfoKHR create_info{
        .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .pNext  = nullptr, .createFlags = 0, .buffer = tlas.buffer.buffer, .offset = 0,
        .size   = sizes.accelerationStructureSize,
        .type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, .deviceAddress = 0,
    };
    validation::checkVulkan(ctx.api.createAccelerationStructure(
        ctx.device, &create_info, nullptr, &tlas.handle));

    auto scratch_align = ctx.accel_props.minAccelerationStructureScratchOffsetAlignment;
    scratch = makeBuffer(ctx, sizes.buildScratchSize + scratch_align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, DEVICE_LOCAL, true);

    VkAccelerationStructureDeviceAddressInfoKHR addr_info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .pNext = nullptr, .accelerationStructure = tlas.handle,
    };
    tlas.address = ctx.api.getAccelStructAddress(ctx.device, &addr_info);
}

// Records a TLAS (re)build from `count` instances into the existing tlas.handle.
void recordTlasBuild(RayTracingContext& ctx, VkCommandBuffer cmd, AccelStruct& tlas,
    RtBuffer& scratch, VkDeviceAddress instances_addr, uint32_t count)
{
    VkAccelerationStructureGeometryKHR geometry{};
    auto build_info = tlasBuildInfo(geometry, instances_addr);
    build_info.dstAccelerationStructure = tlas.handle;
    auto scratch_align = ctx.accel_props.minAccelerationStructureScratchOffsetAlignment;
    build_info.scratchData.deviceAddress = alignUp(scratch.address, scratch_align);

    VkAccelerationStructureBuildRangeInfoKHR range{
        .primitiveCount = count, .primitiveOffset = 0, .firstVertex = 0, .transformOffset = 0,
    };
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
    ctx.api.cmdBuildAccelerationStructures(cmd, 1, &build_info, &p_range);
}

// ---- Ray-tracing pipeline + SBT ---------------------------------------------------

void createRtPipeline(RayTracingContext& ctx)
{
    // Descriptor set layout: TLAS + display image + accumulator + denoiser G-buffer (all written
    // by the raygen except the TLAS, which the closest-hit also reads).
    VkDescriptorSetLayoutBinding bindings[4] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
          .pImmutableSamplers = nullptr },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
          .pImmutableSamplers = nullptr },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
          .pImmutableSamplers = nullptr },
        { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
          .pImmutableSamplers = nullptr },
    };
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .bindingCount = 4, .pBindings = bindings,
    };
    validation::checkVulkan(vkCreateDescriptorSetLayout(
        ctx.device, &set_info, nullptr, &ctx.rt_set_layout));

    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                    | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        .offset = 0,
        .size = sizeof(RtPushConstants),
    };
    VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &ctx.rt_set_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
    };
    validation::checkVulkan(vkCreatePipelineLayout(
        ctx.device, &layout_info, nullptr, &ctx.rt_pipeline_layout));

    // Compile the RT slang module (raygen/miss/closest-hit). Match pipeline.cpp's behavior
    // of running the compile with the shader directory as the working directory.
    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(getDefaultShaderPath());
    auto builder = ShaderBuilder::make();
    ShaderCompileParams params{
        .module_path = "shaders/pathtrace.slang",
        .entrypoints = { "raygenMain", "missMain", "closestHitMain", "sphereIntersect" },
        .specializations = {},
    };
    auto stages = builder.compileModule(ctx.device, params);
    std::filesystem::current_path(orig_path);

    if (stages.size() != 4)
    {
        spdlog::error("pathtrace.slang: expected 4 RT stages, got {}", stages.size());
    }

    // Group each stage: raygen (general), miss (general), then a PROCEDURAL hit group pairing the
    // closest-hit with the sphere intersection shader (AABB geometry, not triangles).
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
    auto general_group = [](uint32_t shader) {
        return VkRayTracingShaderGroupCreateInfoKHR{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .pNext = nullptr,
            .type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = shader,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
            .pShaderGroupCaptureReplayHandle = nullptr,
        };
    };
    uint32_t raygen_idx = VK_SHADER_UNUSED_KHR, miss_idx = VK_SHADER_UNUSED_KHR,
             hit_idx = VK_SHADER_UNUSED_KHR, isect_idx = VK_SHADER_UNUSED_KHR;
    for (uint32_t i = 0; i < stages.size(); ++i)
    {
        switch (stages[i].stage)
        {
            case VK_SHADER_STAGE_RAYGEN_BIT_KHR:       raygen_idx = i; break;
            case VK_SHADER_STAGE_MISS_BIT_KHR:         miss_idx = i;   break;
            case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:  hit_idx = i;    break;
            case VK_SHADER_STAGE_INTERSECTION_BIT_KHR: isect_idx = i;  break;
            default: break;
        }
    }
    groups.push_back(general_group(raygen_idx));
    groups.push_back(general_group(miss_idx));
    groups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .pNext = nullptr,
        .type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
        .generalShader = VK_SHADER_UNUSED_KHR,
        .closestHitShader = hit_idx,
        .anyHitShader = VK_SHADER_UNUSED_KHR,
        .intersectionShader = isect_idx,
        .pShaderGroupCaptureReplayHandle = nullptr,
    });

    VkRayTracingPipelineCreateInfoKHR pipeline_info{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext = nullptr, .flags = 0,
        .stageCount = static_cast<uint32_t>(stages.size()), .pStages = stages.data(),
        .groupCount = static_cast<uint32_t>(groups.size()), .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = ctx.max_recursion,
        .pLibraryInfo = nullptr,
        .pLibraryInterface = nullptr,
        .pDynamicState = nullptr,
        .layout = ctx.rt_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };
    validation::checkVulkan(ctx.api.createRayTracingPipelines(ctx.device,
        VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &ctx.rt_pipeline));

    for (auto& stage : stages) { vkDestroyShaderModule(ctx.device, stage.module, nullptr); }

    // ---- Shader binding table (multi-material) ----
    // Layout: [raygen][miss][hit_0 .. hit_{N-1}]. Each hit record = the closest-hit group handle
    // followed by that material's MaterialData shader record, so the hit stride must fit both.
    uint32_t handle_size = ctx.rt_props.shaderGroupHandleSize;
    uint32_t handle_aligned = static_cast<uint32_t>(
        alignUp(handle_size, ctx.rt_props.shaderGroupHandleAlignment));
    uint32_t base_align = ctx.rt_props.shaderGroupBaseAlignment;
    uint32_t group_count = static_cast<uint32_t>(groups.size());
    uint32_t material_count = static_cast<uint32_t>(ctx.materials.size());
    ctx.shader_handle_size = handle_size;

    uint32_t hit_stride = static_cast<uint32_t>(
        alignUp(handle_size + sizeof(MaterialData), ctx.rt_props.shaderGroupHandleAlignment));

    ctx.raygen_region.stride = alignUp(handle_aligned, base_align);
    ctx.raygen_region.size   = ctx.raygen_region.stride;
    ctx.miss_region.stride   = handle_aligned;
    ctx.miss_region.size     = alignUp(handle_aligned, base_align);
    ctx.hit_region.stride    = hit_stride;
    // Region size covers all material records; align up to the base alignment so the region is
    // self-contained (nothing follows it, but keep the invariant that regions are base-aligned).
    ctx.hit_region.size      = alignUp(VkDeviceSize(hit_stride) * material_count, base_align);

    std::vector<uint8_t> handles(group_count * handle_size);
    validation::checkVulkan(ctx.api.getShaderGroupHandles(ctx.device, ctx.rt_pipeline,
        0, group_count, handles.size(), handles.data()));

    VkDeviceSize sbt_size = ctx.raygen_region.size + ctx.miss_region.size + ctx.hit_region.size;
    ctx.sbt_buffer = makeBuffer(ctx, sbt_size,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        HOST_VISIBLE, true);
    ctx.sbt_hit_records_offset = ctx.raygen_region.size + ctx.miss_region.size;

    std::vector<uint8_t> sbt(sbt_size, 0);
    auto handle_at = [&](uint32_t group){ return handles.data() + group * handle_size; };
    std::memcpy(sbt.data(), handle_at(0), handle_size);                         // raygen (group 0)
    std::memcpy(sbt.data() + ctx.raygen_region.size, handle_at(1), handle_size);// miss   (group 1)
    // One hit record per material: the (single) closest-hit group handle + the material's data.
    for (uint32_t m = 0; m < material_count; ++m)
    {
        uint8_t* rec = sbt.data() + ctx.sbt_hit_records_offset + VkDeviceSize(m) * hit_stride;
        std::memcpy(rec, handle_at(2), handle_size);                    // closest-hit group (group 2)
        std::memcpy(rec + handle_size, &ctx.materials[m], sizeof(MaterialData)); // shader record data
    }
    uploadBuffer(ctx.device, ctx.sbt_buffer, sbt.data(), sbt_size);

    VkDeviceAddress base = ctx.sbt_buffer.address;
    ctx.raygen_region.deviceAddress = base;
    ctx.miss_region.deviceAddress   = base + ctx.raygen_region.size;
    ctx.hit_region.deviceAddress    = base + ctx.sbt_hit_records_offset;
    ctx.callable_region = {};

    // Descriptor pool sized for a few frames in flight (TLAS + three storage images per set:
    // display image, shared accumulator, and the denoiser G-buffer).
    VkDescriptorPoolSize rt_pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 24 },
    };
    VkDescriptorPoolCreateInfo rt_pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .maxSets = 8,
        .poolSizeCount = 2, .pPoolSizes = rt_pool_sizes,
    };
    validation::checkVulkan(vkCreateDescriptorPool(
        ctx.device, &rt_pool_info, nullptr, &ctx.rt_pool));
}

// ---- Fullscreen composite pipeline -----------------------------------------------

void createCompositeResources(RayTracingContext& ctx)
{
    ctx.composite_sampler = createSampler(ctx.device, VK_FILTER_LINEAR, false);

    VkDescriptorSetLayoutBinding binding{
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr,
    };
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .bindingCount = 1, .pBindings = &binding,
    };
    validation::checkVulkan(vkCreateDescriptorSetLayout(
        ctx.device, &set_info, nullptr, &ctx.composite_set_layout));

    VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &ctx.composite_set_layout,
        .pushConstantRangeCount = 0, .pPushConstantRanges = nullptr,
    };
    validation::checkVulkan(vkCreatePipelineLayout(
        ctx.device, &layout_info, nullptr, &ctx.composite_pipeline_layout));

    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 };
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .maxSets = 8,
        .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    validation::checkVulkan(vkCreateDescriptorPool(
        ctx.device, &pool_info, nullptr, &ctx.composite_pool));
}

VkPipeline buildCompositePipeline(RayTracingContext& ctx, VkRenderPass render_pass)
{
    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(getDefaultShaderPath());
    auto builder = ShaderBuilder::make();
    ShaderCompileParams params{
        .module_path = "shaders/pathtrace_composite.slang",
        .entrypoints = { "vertexMain", "fragmentMain" },
        .specializations = {},
    };
    auto stages = builder.compileModule(ctx.device, params);
    std::filesystem::current_path(orig_path);

    VkPipelineVertexInputStateCreateInfo vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .vertexBindingDescriptionCount = 0, .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0, .pVertexAttributeDescriptions = nullptr,
    };
    VkPipelineInputAssemblyStateCreateInfo input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    VkViewport viewport{ 0.f, 0.f, float(ctx.extent.width), float(ctx.extent.height), 0.f, 1.f };
    VkRect2D scissor{ {0, 0}, ctx.extent };
    VkPipelineViewportStateCreateInfo viewport_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .viewportCount = 1, .pViewports = &viewport,
        .scissorCount = 1, .pScissors = &scissor,
    };
    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .depthClampEnable = VK_FALSE, .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.f, .depthBiasClamp = 0.f,
        .depthBiasSlopeFactor = 0.f, .lineWidth = 1.f,
    };
    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.f, .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE, .alphaToOneEnable = VK_FALSE,
    };
    // The composite overwrites every pixel; depth test disabled so it ignores the depth buffer.
    VkPipelineDepthStencilStateCreateInfo depth_stencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .depthTestEnable = VK_FALSE, .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_ALWAYS, .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE, .front = {}, .back = {},
        .minDepthBounds = 0.f, .maxDepthBounds = 1.f,
    };
    VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                        | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    VkPipelineColorBlendStateCreateInfo color_blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .logicOpEnable = VK_FALSE, .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1, .pAttachments = &blend_attachment,
        .blendConstants = {0.f, 0.f, 0.f, 0.f},
    };
    VkGraphicsPipelineCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .stageCount = static_cast<uint32_t>(stages.size()), .pStages = stages.data(),
        .pVertexInputState = &vertex_input, .pInputAssemblyState = &input_assembly,
        .pTessellationState = nullptr, .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_stencil, .pColorBlendState = &color_blend,
        .pDynamicState = nullptr, .layout = ctx.composite_pipeline_layout,
        .renderPass = render_pass, .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE, .basePipelineIndex = 0,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateGraphicsPipelines(
        ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    for (auto& stage : stages) { vkDestroyShaderModule(ctx.device, stage.module, nullptr); }
    return pipeline;
}

// ---- Instance-writer compute pipeline --------------------------------------------

// Push constants for pathtrace_aabbs.slang: the AABB output buffer as a buffer-device-address
// pointer (offset 0, 8-byte aligned), then the particle count and sphere radius.
struct AabbWriterPush
{
    VkDeviceAddress aabbs;     // BDA pointer to the AABB output buffer (offset 0, 8-byte aligned)
    VkDeviceAddress positions; // BDA pointer to the interop positions (offset 8)
    uint32_t count;
    float radius;
};

void createAabbWriter(RayTracingContext& ctx)
{
    // The writer has NO descriptors: both the AABB output and the positions input ride the push
    // constants as buffer-device-address pointers (see PushConstants in pathtrace_aabbs.slang), so
    // neither is bound and neither is capped by maxStorageBufferRange. The pipeline layout is thus
    // just the push-constant range.
    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(AabbWriterPush),
    };
    VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 0, .pSetLayouts = nullptr,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
    };
    validation::checkVulkan(vkCreatePipelineLayout(
        ctx.device, &layout_info, nullptr, &ctx.iw_pipeline_layout));

    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(getDefaultShaderPath());
    auto builder = ShaderBuilder::make();
    ShaderCompileParams params{
        .module_path = "shaders/pathtrace_aabbs.slang",
        .entrypoints = { "writeAabbsMain" }, .specializations = {},
    };
    auto stages = builder.compileModule(ctx.device, params);
    std::filesystem::current_path(orig_path);
    if (stages.size() != 1)
    {
        spdlog::error("pathtrace_aabbs.slang: expected 1 compute stage, got {}", stages.size());
    }

    VkComputePipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .stage = stages[0],
        .layout = ctx.iw_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE, .basePipelineIndex = 0,
    };
    validation::checkVulkan(vkCreateComputePipelines(
        ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &ctx.iw_pipeline));
    vkDestroyShaderModule(ctx.device, stages[0].module, nullptr);
}

// ---- À-trous denoiser compute pipeline -------------------------------------------

// Push constants for pathtrace_atrous.slang (40 bytes). Must match its PushConstants.
struct AtrousPush
{
    int32_t dims_x;
    int32_t dims_y;
    int32_t step_width;
    float c_phi;
    float n_phi;
    float p_phi;
    uint32_t tonemap;
    // Backdrop colour (background * intensity); the final pass shows it on primary-miss pixels.
    float sky_r, sky_g, sky_b;
};

void createAtrousPipeline(RayTracingContext& ctx)
{
    // Three storage images: in_color, gbuffer, out_color.
    VkDescriptorSetLayoutBinding bindings[3] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr },
    };
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .bindingCount = 3, .pBindings = bindings,
    };
    validation::checkVulkan(vkCreateDescriptorSetLayout(
        ctx.device, &set_info, nullptr, &ctx.atrous_set_layout));

    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(AtrousPush),
    };
    VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &ctx.atrous_set_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
    };
    validation::checkVulkan(vkCreatePipelineLayout(
        ctx.device, &layout_info, nullptr, &ctx.atrous_pipeline_layout));

    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(getDefaultShaderPath());
    auto builder = ShaderBuilder::make();
    ShaderCompileParams params{
        .module_path = "shaders/pathtrace_atrous.slang",
        .entrypoints = { "atrousMain" }, .specializations = {},
    };
    auto stages = builder.compileModule(ctx.device, params);
    std::filesystem::current_path(orig_path);
    if (stages.size() != 1)
    {
        spdlog::error("pathtrace_atrous.slang: expected 1 compute stage, got {}", stages.size());
    }

    VkComputePipelineCreateInfo pipeline_info{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .stage = stages[0],
        .layout = ctx.atrous_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE, .basePipelineIndex = 0,
    };
    validation::checkVulkan(vkCreateComputePipelines(
        ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &ctx.atrous_pipeline));
    vkDestroyShaderModule(ctx.device, stages[0].module, nullptr);

    // ATROUS_PASSES sets per frame, each with 3 storage images.
    uint32_t set_count = RayTracingContext::ATROUS_PASSES * RayTracingContext::FRAMES;
    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * set_count };
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .maxSets = set_count,
        .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    validation::checkVulkan(vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &ctx.atrous_pool));
}

// Allocate the per-frame descriptor sets from their pools (bindings are written later, in
// createFrameResources for the storage image and bindScene for the TLAS/instances).
void allocateDescriptorSets(RayTracingContext& ctx)
{
    auto alloc = [&](VkDescriptorPool pool, VkDescriptorSetLayout layout, VkDescriptorSet* out) {
        std::vector<VkDescriptorSetLayout> layouts(RayTracingContext::FRAMES, layout);
        VkDescriptorSetAllocateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr, .descriptorPool = pool,
            .descriptorSetCount = RayTracingContext::FRAMES, .pSetLayouts = layouts.data(),
        };
        validation::checkVulkan(vkAllocateDescriptorSets(ctx.device, &info, out));
    };
    ctx.rt_sets.resize(RayTracingContext::FRAMES);
    ctx.composite_sets.resize(RayTracingContext::FRAMES);
    alloc(ctx.rt_pool, ctx.rt_set_layout, ctx.rt_sets.data());
    alloc(ctx.composite_pool, ctx.composite_set_layout, ctx.composite_sets.data());
    // The AABB writer has no descriptor set (positions + AABBs are BDA push-constant pointers).

    // À-trous denoiser: ATROUS_PASSES sets per frame (bindings written in createFrameResources).
    uint32_t atrous_count = RayTracingContext::ATROUS_PASSES * RayTracingContext::FRAMES;
    std::vector<VkDescriptorSetLayout> atrous_layouts(atrous_count, ctx.atrous_set_layout);
    VkDescriptorSetAllocateInfo atrous_alloc{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr, .descriptorPool = ctx.atrous_pool,
        .descriptorSetCount = atrous_count, .pSetLayouts = atrous_layouts.data(),
    };
    ctx.atrous_sets.resize(atrous_count);
    validation::checkVulkan(vkAllocateDescriptorSets(ctx.device, &atrous_alloc, ctx.atrous_sets.data()));
}

} // namespace

// ---- LOD grid-aggregation compute pipelines ---------------------------------------

// Push constants for pathtrace_lod_scatter.slang: positions BDA pointer (offset 0), then the
// particle count and grid resolution. The per-cell count buffer is a descriptor (binding 0).
struct LodScatterPush { VkDeviceAddress positions; uint32_t count; uint32_t gridN; };
// Push constants for pathtrace_lod_emit.slang: AABB output BDA pointer (offset 0), grid
// resolution, sphere radius. Cell counts (binding 0) and the global counter (binding 1) are
// descriptors.
struct LodEmitPush { VkDeviceAddress aabbs; uint32_t gridN; float radius; };

void RayTracingContext::createLodPipelines()
{
    // One-binding set for scatter (cell counts), two-binding set for emit (cell counts + counter).
    auto make_set_layout = [&](uint32_t binding_count) {
        std::vector<VkDescriptorSetLayoutBinding> b(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            b[i] = VkDescriptorSetLayoutBinding{
                .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr };
        }
        VkDescriptorSetLayoutCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .bindingCount = binding_count, .pBindings = b.data() };
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        validation::checkVulkan(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));
        return layout;
    };
    lod_scatter_set_layout = make_set_layout(1);
    lod_emit_set_layout    = make_set_layout(2);

    auto make_pipeline = [&](VkDescriptorSetLayout set_layout, uint32_t push_size,
                             const char* module, const char* entry,
                             VkPipelineLayout& out_layout, VkPipeline& out_pipe) {
        VkPushConstantRange range{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size };
        VkPipelineLayoutCreateInfo li{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &set_layout,
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
    make_pipeline(lod_scatter_set_layout, sizeof(LodScatterPush),
        "shaders/pathtrace_lod_scatter.slang", "scatterMain", lod_scatter_layout, lod_scatter_pipeline);
    make_pipeline(lod_emit_set_layout, sizeof(LodEmitPush),
        "shaders/pathtrace_lod_emit.slang", "emitMain", lod_emit_layout, lod_emit_pipeline);

    // One set of each (the aggregate runs in an internal one-shot submit, not per-frame-in-flight).
    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }; // 1 (scatter) + 2 (emit)
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr, .flags = 0,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    validation::checkVulkan(vkCreateDescriptorPool(device, &pool_info, nullptr, &lod_desc_pool));

    VkDescriptorSetLayout layouts[2] = { lod_scatter_set_layout, lod_emit_set_layout };
    VkDescriptorSet sets[2] = {};
    VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr,
        .descriptorPool = lod_desc_pool, .descriptorSetCount = 2, .pSetLayouts = layouts };
    validation::checkVulkan(vkAllocateDescriptorSets(device, &ai, sets));
    lod_scatter_set = sets[0];
    lod_emit_set    = sets[1];
}

RayTracingContext RayTracingContext::make(VkDevice device, VkPhysicalDevice gpu,
    VkPhysicalDeviceMemoryProperties mem_props, SubmitFn submit,
    uint32_t subdiv, uint32_t max_recursion, std::vector<MaterialData> materials)
{
    RayTracingContext ctx{};
    ctx.device = device;
    ctx.physical_device = gpu;
    ctx.mem_props = mem_props;
    ctx.submit = std::move(submit);
    ctx.api = RayTracingApi::load(device);
    ctx.max_recursion = max_recursion;

    // Seed the multi-material SBT. At least one material always exists (material 0 = the particle
    // surface, whose albedo bindScene later overwrites with the view color).
    ctx.materials = std::move(materials);
    if (ctx.materials.empty()) { ctx.materials.push_back(MaterialData{}); }

    // Query RT pipeline + acceleration-structure properties (SBT strides, scratch align).
    ctx.rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    ctx.accel_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    ctx.rt_props.pNext = &ctx.accel_props;
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &ctx.rt_props, .properties = {},
    };
    vkGetPhysicalDeviceProperties2(gpu, &props2);

    // GPU-timestamp timing pool for the HUD/CSV (FRAMES*TS_PER_FRAME timestamps; see the header).
    // Disabled if the device reports a 0 timestamp period (no timestamp support), leaving timings at 0.
    ctx.timestamp_period = props2.properties.limits.timestampPeriod;
    if (ctx.timestamp_period > 0.f)
    {
        VkQueryPoolCreateInfo qinfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = FRAMES * TS_PER_FRAME,
            .pipelineStatistics = 0,
        };
        validation::checkVulkan(vkCreateQueryPool(device, &qinfo, nullptr, &ctx.timing_pool));
        vkResetQueryPool(device, ctx.timing_pool, 0, FRAMES * TS_PER_FRAME);
    }

    // subdiv is no longer used: spheres are analytic procedural AABBs (no tessellated mesh), so the
    // BLAS is built per-view in bindScene rather than a static icosphere here.
    (void)subdiv;
    createRtPipeline(ctx);
    createCompositeResources(ctx);
    createAabbWriter(ctx);
    createAtrousPipeline(ctx);
    ctx.createLodPipelines();
    allocateDescriptorSets(ctx);

    spdlog::info("Path tracing ready: procedural AABB spheres; scene bound per view");
    return ctx;
}

void RayTracingContext::createFrameResources(VkExtent2D new_extent, VkRenderPass render_pass)
{
    extent = new_extent;

    // Helper: allocate a device-local StorageImage of the given format/usage at the current extent.
    auto makeStorageImage = [&](VkFormat format, VkImageUsageFlags usage) {
        ImageParams params{
            .type = VK_IMAGE_TYPE_2D, .format = format,
            .extent = { extent.width, extent.height, 1 },
            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage, .levels = 1,
        };
        StorageImage si{};
        si.image = createImage(device, physical_device, params);
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, si.image, &req);
        si.memory = allocateMemory(device, mem_props, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        validation::checkVulkan(vkBindImageMemory(device, si.image, si.memory, 0));
        si.view = createImageView(device, si.image, params, VK_IMAGE_ASPECT_COLOR_BIT);
        return si;
    };

    // Storage images (one per frame in flight).
    storage_images.assign(FRAMES, {});
    for (uint32_t i = 0; i < FRAMES; ++i)
    {
        ImageParams params{
            .type = VK_IMAGE_TYPE_2D, .format = storage_format,
            .extent = { extent.width, extent.height, 1 },
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .levels = 1,
        };
        auto& si = storage_images[i];
        si.image = createImage(device, physical_device, params);
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, si.image, &req);
        si.memory = allocateMemory(device, mem_props, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        validation::checkVulkan(vkBindImageMemory(device, si.image, si.memory, 0));
        si.view = createImageView(device, si.image, params, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Persistent HDR accumulator (single image, shared across frames in flight). Higher precision
    // (RGBA32F) than the display images because it holds a running mean over many frames. It is
    // never discarded between frames, so it is transitioned to GENERAL once here and kept there;
    // recordTrace only inserts a read/write barrier. A fresh image resets accumulation on resize.
    {
        ImageParams params{
            .type = VK_IMAGE_TYPE_2D, .format = accum_format,
            .extent = { extent.width, extent.height, 1 },
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
            .levels = 1,
        };
        accum_image.image = createImage(device, physical_device, params);
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, accum_image.image, &req);
        accum_image.memory = allocateMemory(device, mem_props, req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        validation::checkVulkan(vkBindImageMemory(device, accum_image.image, accum_image.memory, 0));
        accum_image.view = createImageView(device, accum_image.image, params, VK_IMAGE_ASPECT_COLOR_BIT);

        // One-time UNDEFINED -> GENERAL transition (contents are discarded; the first traced frame
        // runs with accum_frame == 0, which overwrites rather than reads the buffer).
        submit([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier to_general{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
                .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = accum_image.image,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &to_general);
        });
    }

    // Denoiser images (per frame): the first-hit G-buffer written by the raygen, and two ping-pong
    // scratch buffers for the à-trous passes. All are used only in GENERAL, so transition once here
    // (contents are regenerated every frame). If denoising is off these simply go unused.
    gbuffer_images.assign(FRAMES, {});
    denoise_ping[0].assign(FRAMES, {});
    denoise_ping[1].assign(FRAMES, {});
    for (uint32_t i = 0; i < FRAMES; ++i)
    {
        gbuffer_images[i] = makeStorageImage(gbuffer_format, VK_IMAGE_USAGE_STORAGE_BIT);
        denoise_ping[0][i] = makeStorageImage(storage_format, VK_IMAGE_USAGE_STORAGE_BIT);
        denoise_ping[1][i] = makeStorageImage(storage_format, VK_IMAGE_USAGE_STORAGE_BIT);
    }
    submit([&](VkCommandBuffer cmd) {
        std::vector<VkImageMemoryBarrier> barriers;
        auto add = [&](VkImage img) {
            barriers.push_back(VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
                .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            });
        };
        for (uint32_t i = 0; i < FRAMES; ++i)
        {
            add(gbuffer_images[i].image);
            add(denoise_ping[0][i].image);
            add(denoise_ping[1][i].image);
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
    });

    // (Re)build the composite pipeline for this extent/render pass.
    if (composite_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, composite_pipeline, nullptr);
    }
    composite_pipeline = buildCompositePipeline(*this, render_pass);

    // Point the storage-image (RT binding 1) and composite (binding 0) descriptors at the
    // new images. The RT TLAS binding (0) is written by bindScene and survives resizes.
    for (uint32_t i = 0; i < FRAMES; ++i)
    {
        VkDescriptorImageInfo storage_info{
            .sampler = VK_NULL_HANDLE, .imageView = storage_images[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorImageInfo accum_info{
            .sampler = VK_NULL_HANDLE, .imageView = accum_image.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorImageInfo gbuffer_info{
            .sampler = VK_NULL_HANDLE, .imageView = gbuffer_images[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        VkDescriptorImageInfo sampled_info{
            .sampler = composite_sampler, .imageView = storage_images[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet writes[4] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = rt_sets[i], .dstBinding = 1, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &storage_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = rt_sets[i], .dstBinding = 2, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &accum_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = rt_sets[i], .dstBinding = 3, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &gbuffer_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = composite_sets[i], .dstBinding = 0, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &sampled_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
        };
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }

    // Wire the à-trous descriptor sets: ATROUS_PASSES per frame, chaining the color buffers
    // accum -> ping0 -> ping1 -> display image, with the G-buffer bound as the guide in each pass.
    for (uint32_t i = 0; i < FRAMES; ++i)
    {
        // Per-pass (in_color, out_color); binding 1 is always this frame's G-buffer guide.
        VkImageView chain_in[ATROUS_PASSES]  = {
            accum_image.view, denoise_ping[0][i].view, denoise_ping[1][i].view };
        VkImageView chain_out[ATROUS_PASSES] = {
            denoise_ping[0][i].view, denoise_ping[1][i].view, storage_images[i].view };
        for (uint32_t pass = 0; pass < ATROUS_PASSES; ++pass)
        {
            VkDescriptorSet set = atrous_sets[i * ATROUS_PASSES + pass];
            VkDescriptorImageInfo in_info{ VK_NULL_HANDLE, chain_in[pass], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo guide_info{ VK_NULL_HANDLE, gbuffer_images[i].view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo out_info{ VK_NULL_HANDLE, chain_out[pass], VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet aw[3] = {
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
                  .dstSet = set, .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  .pImageInfo = &in_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
                  .dstSet = set, .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  .pImageInfo = &guide_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
                  .dstSet = set, .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1,
                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  .pImageInfo = &out_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
            };
            vkUpdateDescriptorSets(device, 3, aw, 0, nullptr);
        }
    }
}

void RayTracingContext::updateMaterial(uint32_t index)
{
    if (index >= materials.size() || sbt_buffer.memory == VK_NULL_HANDLE) { return; }
    // The material data sits right after the group handle in this material's hit record. The SBT
    // buffer is host-visible + coherent, so a plain mapped write suffices (setup-time, no in-flight
    // frames). We map only this record's data slice.
    VkDeviceSize offset = sbt_hit_records_offset
        + VkDeviceSize(index) * hit_region.stride + shader_handle_size;
    void* mapped = nullptr;
    validation::checkVulkan(vkMapMemory(device, sbt_buffer.memory, offset,
        sizeof(MaterialData), 0, &mapped));
    std::memcpy(mapped, &materials[index], sizeof(MaterialData));
    vkUnmapMemory(device, sbt_buffer.memory);
}

void RayTracingContext::bindScene(VkDeviceAddress positions, uint32_t count, float radius, glm::vec4 color)
{
    position_address = positions;
    particle_count  = count;
    particle_radius = radius;
    particle_color  = color;

    // Material 0 is the particle surface: adopt the view color as its albedo and push it into the
    // SBT hit record. Other registered materials keep their configured data.
    if (!materials.empty())
    {
        materials[0].albedo = glm::vec3(color);
        updateMaterial(0);
    }

    // Per-particle AABB buffer (written by the compute writer, read by the BLAS build). Reached by
    // device address (want_address below), NOT a STORAGE descriptor, so its size is not capped by
    // maxStorageBufferRange; it needs only AS_BUILD_INPUT (BLAS read) and a device address (BDA store
    // + BLAS build input). One shared copy (see raytracing.hpp).
    //
    // LOD aggregates particles into <= min(N^3, P) occupied-cell spheres, so both the AABB buffer
    // and the BLAS are sized to that bound (smaller than P at large N). Per-particle mode sizes to P.
    uint32_t geom_prims = count;
    if (lod_cells > 0)
    {
        uint64_t cells = uint64_t(lod_cells) * lod_cells * lod_cells;
        lod_max_cells = static_cast<uint32_t>(std::min<uint64_t>(cells, count));
        geom_prims = lod_max_cells;

        // Per-cell occupancy counts (one uint each) + the emitted-primitive counter (host-readable).
        lod_cellcount_buffer = makeBuffer(*this, VkDeviceSize(cells) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE_LOCAL, false);
        lod_counter_buffer = makeBuffer(*this, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_VISIBLE, true);
    }

    VkDeviceSize aabb_size = VkDeviceSize(geom_prims) * sizeof(VkAabbPositionsKHR);
    aabb_buffer = makeBuffer(*this, aabb_size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        DEVICE_LOCAL, true);

    // Split into BLAS chunks of <= maxPrimitiveCount AABBs (env override for testing the multi-chunk
    // path at small N). maxPrimitiveCount is a per-BLAS limit; chunking beats it. Chunk size is packed
    // into the push constant (cam_pos.w) so the intersection shader can turn its local PrimitiveIndex
    // back into a global one.
    blas_chunk_prims = accel_props.maxPrimitiveCount > 0
        ? static_cast<uint32_t>(std::min<uint64_t>(accel_props.maxPrimitiveCount, geom_prims))
        : geom_prims;
    // LOD mode requires exactly one BLAS chunk: recordLodUpdate() passes the per-frame `occupied`
    // count as override_prims to recordBlasBuildChunks(), which applies it to EVERY chunk. If the env
    // override shrank blas_chunk_prims below lod_max_cells here, num_chunks would become > 1 and
    // chunk c>0 would read past the emitted primitive list / AABB buffer. So the override only applies
    // in per-particle mode, where blas_chunk_prims is already <= maxPrimitiveCount as computed above.
    if (lod_cells == 0)
    {
        if (const char* env = std::getenv("MIMIR_PT_BLAS_CHUNK"))
        {
            uint32_t override_prims = static_cast<uint32_t>(std::strtoull(env, nullptr, 10));
            if (override_prims > 0 && override_prims < blas_chunk_prims) { blas_chunk_prims = override_prims; }
        }
    }
    else if (std::getenv("MIMIR_PT_BLAS_CHUNK") != nullptr)
    {
        spdlog::info("Path tracing: MIMIR_PT_BLAS_CHUNK ignored in LOD mode (single chunk required)");
    }

    // One BLAS per chunk (rebuilt each frame), then one TLAS instance per chunk (identity transform,
    // instanceCustomIndex = chunk index). BLAS device addresses are stable across rebuilds, so the
    // instances are written once here.
    createDynamicBlasChunks(*this, aabb_buffer.address, geom_prims);
    uint32_t num_chunks = static_cast<uint32_t>(scene_blas.size());
    spdlog::info("Path tracing: {} particles in {} BLAS chunk(s) of up to {} prims",
        count, num_chunks, blas_chunk_prims);
    if (lod_cells > 0)
    {
        spdlog::info("Path tracing: LOD {}^3 grid, up to {} occupied-cell primitives (from {} particles)",
            lod_cells, lod_max_cells, count);
    }

    // BLAS refit cadence: full rebuild every rebuild_interval dirty frames, refit in between (see the
    // members in raytracing.hpp). At ~5*10^8 AABBs the full rebuild is the frame's dominant cost, so a
    // larger interval trades traversal quality for speed. Env override MIMIR_PT_REBUILD_INTERVAL (0/1
    // forces a full rebuild every frame = the old behaviour).
    if (const char* env = std::getenv("MIMIR_PT_REBUILD_INTERVAL"))
    {
        rebuild_interval = static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
    }
    if (rebuild_interval <= 1)
    {
        spdlog::info("Path tracing: BLAS refit disabled (full rebuild every frame)");
    }
    else
    {
        spdlog::info("Path tracing: BLAS refit on, full rebuild every {} frames", rebuild_interval);
    }

    tlas_instance_buffer = makeBuffer(*this,
        VkDeviceSize(num_chunks) * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, HOST_VISIBLE, true);
    std::vector<VkAccelerationStructureInstanceKHR> insts(num_chunks);
    for (uint32_t c = 0; c < num_chunks; ++c)
    {
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform = { .matrix = { {1.f, 0.f, 0.f, 0.f},
                                       {0.f, 1.f, 0.f, 0.f},
                                       {0.f, 0.f, 1.f, 0.f} } }; // identity 3x4
        inst.instanceCustomIndex = c & 0xFFFFFFu;    // chunk index -> InstanceID() in the shader
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0; // material 0 (the particle surface)
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = scene_blas[c].address;
        insts[c] = inst;
    }
    uploadBuffer(device, tlas_instance_buffer, insts.data(),
        insts.size() * sizeof(VkAccelerationStructureInstanceKHR));

    createDynamicTlas(*this, scene_tlas, tlas_scratch, tlas_instance_buffer.address, num_chunks);

    // Point each RT set's TLAS binding (0) at the shared TLAS. Both the AABB buffer and the positions
    // ride the AABB-writer push constants as BDA pointers (no descriptors), so the writer has no set to
    // wire here. The RT storage-image/accum bindings (1,2) were written by createFrameResources.
    for (uint32_t f = 0; f < FRAMES; ++f)
    {
        VkWriteDescriptorSetAccelerationStructureKHR as_write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr, .accelerationStructureCount = 1,
            .pAccelerationStructures = &scene_tlas.handle,
        };
        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &as_write,
            .dstSet = rt_sets[f], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .pImageInfo = nullptr, .pBufferInfo = nullptr, .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    if (lod_cells > 0)
    {
        VkDescriptorBufferInfo cc{ .buffer = lod_cellcount_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
        VkDescriptorBufferInfo gc{ .buffer = lod_counter_buffer.buffer,  .offset = 0, .range = VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[3] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = lod_scatter_set,
              .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
              .pBufferInfo = &cc, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = lod_emit_set,
              .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
              .pBufferInfo = &cc, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = lod_emit_set,
              .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
              .pBufferInfo = &gc, .pTexelBufferView = nullptr },
        };
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }

    // Populate the AABBs and build the BLAS + TLAS once (positions are already valid at bind time),
    // so the first rendered frame has a coherent AS even before its own update.
    submit([&](VkCommandBuffer cmd) {
        recordUpdateScene(cmd, 0);
    });

    scene_bound = true;
    spdlog::info("Path tracing scene bound: {} AABB spheres, radius {:.4f}", count, radius);
}

void RayTracingContext::recordUpdateScene(VkCommandBuffer cmd, uint32_t frame_idx, bool rebuild)
{
    if (particle_count == 0) { return; }

    // The first call must build (an UPDATE refit needs a prior full build as its source); after that
    // the caller's `rebuild` flag decides. A rebuilt-from-scratch BLAS over ~5*10^8 AABBs is the frame's
    // dominant cost, so we do as little as the scene allows.
    bool must_build = rebuild || !accel_ever_built;

    if (lod_cells > 0)
    {
        if (!must_build)
        {
            // Same skip bookkeeping as the per-particle path: reuse the existing AS, stamp ~0 build.
            stat_skips++;
            slot_build_mode[frame_idx] = BlasBuild::Skip;
            if (timing_pool != VK_NULL_HANDLE)
            {
                vkCmdResetQueryPool(cmd, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME);
                uint32_t base = frame_idx * TS_PER_FRAME;
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,    timing_pool, base + 0);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 1);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 3);
            }
            return;
        }
        recordLodUpdate(cmd, frame_idx);
        return;
    }

    if (!must_build)
    {
        stat_skips++;
        slot_build_mode[frame_idx] = BlasBuild::Skip;
        // Positions unchanged since the last built frame (paused / no new sim step): the shared
        // BLAS/TLAS are still valid, so skip the AABB-writer + AS (re)build entirely and let the trace
        // reuse them (the accumulator keeps converging on the static scene). No writes here means no
        // write-after-read hazard on the shared resources, so the cross-frame serialize barrier is not
        // needed. Still reset this frame's timestamps (recordTrace writes the trace slots, which must
        // be reset first) and stamp every build sub-phase as ~0 by placing marks +0..+3 adjacently.
        if (timing_pool != VK_NULL_HANDLE)
        {
            vkCmdResetQueryPool(cmd, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME);
            uint32_t base = frame_idx * TS_PER_FRAME;
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,    timing_pool, base + 0);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 1);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 3);
        }
        return;
    }

    // Cross-frame serialization: the AABB buffer, BLAS, TLAS, and scratch are single shared copies, so
    // this frame's AABB-writer (overwrites the AABBs) and AS builds (overwrite the BLAS/TLAS + scratch)
    // must not start until the PREVIOUS frame's trace has finished reading the TLAS -- a write-after-
    // read hazard on shared resources. Same queue + submission order means an execution dependency from
    // ray tracing to compute/build here orders us after all prior traces. This is the pipelining cost of
    // not triple-buffering the (potentially many-GiB) AABB buffer; on the first frame there is no prior
    // trace, so the barrier is a no-op.
    VkMemoryBarrier serialize{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT
                       | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                       | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &serialize, 0, nullptr, 0, nullptr);

    // Reset this frame's TS_PER_FRAME timestamps (previous readings for frame_idx were already read
    // back in renderFrame) and stamp the start of the build phase (before the AABB writer). The
    // per-phase boundary marks (+1 after AABB, +2 after BLAS, +3 after TLAS) are stamped below, so
    // readTimings can attribute the build time to its three sub-phases.
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 0);
    }

    // 1) AABB writer: fill the shared AABB buffer from the live positions (one sphere per particle).
    // Both buffers are BDA pointers in the push constants, so there is no descriptor set to bind.
    AabbWriterPush push{
        .aabbs = aabb_buffer.address,
        .positions = position_address,
        .count = particle_count, .radius = particle_radius,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, iw_pipeline);
    vkCmdPushConstants(cmd, iw_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), &push);
    vkCmdDispatch(cmd, (particle_count + 63) / 64, 1, 1);

    // Barrier: compute writes -> BLAS build reads the AABB buffer.
    VkMemoryBarrier to_build{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &to_build, 0, nullptr, 0, nullptr);

    // Sub-phase boundary: AABB writer done. The to_build barrier already made the compute writes
    // complete before the BLAS build, so a BOTTOM_OF_PIPE mark here fences the AABB writer's time.
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 1);
    }

    // 2) (Re)build every chunk BLAS from its AABB slice (serialized on the shared scratch inside).
    // Refit in place most frames; force a full rebuild on the first build and every rebuild_interval
    // dirty frames (rebuild_interval <= 1 disables refit). The OU walk keeps particles bounded, so the
    // topology from the last full rebuild stays a good fit between forced rebuilds.
    bool do_refit = accel_ever_built && rebuild_interval > 1
                  && (frames_since_full_rebuild + 1) < rebuild_interval;
    recordBlasBuildChunks(*this, cmd, aabb_buffer.address, do_refit);
    accel_ever_built = true;
    frames_since_full_rebuild = do_refit ? frames_since_full_rebuild + 1 : 0;
    if (do_refit) { stat_refits++; } else { stat_full_rebuilds++; }
    slot_build_mode[frame_idx] = do_refit ? BlasBuild::Refit : BlasBuild::Rebuild;

    // Barrier: BLAS build writes -> TLAS build reads the BLASes (via the instances' AS references).
    VkMemoryBarrier blas_to_tlas{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &blas_to_tlas, 0, nullptr, 0, nullptr);

    // Sub-phase boundary: BLAS build/refit done (fenced by the blas_to_tlas barrier above).
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 2);
    }

    // 3) Rebuild the TLAS over the per-chunk instances (their cached BLAS bounds change as particles move).
    recordTlasBuild(*this, cmd, scene_tlas, tlas_scratch, tlas_instance_buffer.address,
        static_cast<uint32_t>(scene_blas.size()));

    // Barrier: AS build writes -> ray tracing reads the TLAS.
    VkMemoryBarrier to_trace{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &to_trace, 0, nullptr, 0, nullptr);

    // Sub-phase boundary: TLAS rebuild done = end of the whole build phase (fenced by to_trace).
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 3);
    }
}

// LOD path: aggregate particles into occupied-cell spheres on the GPU (clear -> scatter -> emit)
// in an internal one-shot submit, read the emitted count back, then full-rebuild the BLAS/TLAS over
// exactly that many primitives in the frame command buffer. The one-shot's vkQueueWaitIdle also
// serializes against the previous frame's trace (which reads the AABB buffer), so no separate
// cross-frame barrier is needed here. Aggregate time is not itemized in the GPU sub-phase split
// (it runs outside `cmd`); it shows up in the total wall-clock render time.
void RayTracingContext::recordLodUpdate(VkCommandBuffer cmd, uint32_t frame_idx)
{
    // Belt-and-suspenders: LOD assumes a single BLAS chunk (see the MIMIR_PT_BLAS_CHUNK guard in
    // bindScene) because override_prims below is applied to every chunk in recordBlasBuildChunks.
    assert(scene_blas.size() == 1 && "LOD requires a single BLAS chunk");
    bool first_build = !accel_ever_built;

    const uint32_t grid = lod_cells;
    const uint64_t num_cells = uint64_t(grid) * grid * grid;
    const float cell_size = 2.0f / float(grid);
    const float radius = LOD_COVERAGE * cell_size * 0.5f;

    // 1) Aggregate in a blocking one-shot submit.
    submit([&](VkCommandBuffer c) {
        vkCmdFillBuffer(c, lod_cellcount_buffer.buffer, 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(c, lod_counter_buffer.buffer,   0, VK_WHOLE_SIZE, 0u);
        VkMemoryBarrier clr{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &clr, 0, nullptr, 0, nullptr);

        LodScatterPush sp{ .positions = position_address, .count = particle_count, .gridN = grid };
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_scatter_pipeline);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_scatter_layout, 0, 1,
            &lod_scatter_set, 0, nullptr);
        vkCmdPushConstants(c, lod_scatter_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
        vkCmdDispatch(c, (particle_count + 63) / 64, 1, 1);

        VkMemoryBarrier s2e{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &s2e, 0, nullptr, 0, nullptr);

        LodEmitPush ep{ .aabbs = aabb_buffer.address, .gridN = grid, .radius = radius };
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_emit_pipeline);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_emit_layout, 0, 1,
            &lod_emit_set, 0, nullptr);
        vkCmdPushConstants(c, lod_emit_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ep), &ep);
        vkCmdDispatch(c, static_cast<uint32_t>((num_cells + 63) / 64), 1, 1);
    });

    // 2) Read the emitted primitive count (HOST_VISIBLE, coherent).
    uint32_t occupied = 0;
    void* mapped = nullptr;
    validation::checkVulkan(vkMapMemory(device, lod_counter_buffer.memory, 0, sizeof(uint32_t), 0, &mapped));
    std::memcpy(&occupied, mapped, sizeof(uint32_t));
    vkUnmapMemory(device, lod_counter_buffer.memory);
    occupied = std::min(occupied, lod_max_cells);
    lod_prim_count = occupied;

    // 3) Build the AS in the frame command buffer over `occupied` primitives (always a full rebuild).
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 0);
        // Aggregate ran outside `cmd`; mark the AABB sub-phase as ~0 here.
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 1);
    }

    // Barrier: the emit compute shader's writes to aabb_buffer happened inside the one-shot submit
    // above. vkQueueWaitIdle (inside submit()) gives host-domain execution ordering only -- it does
    // not guarantee device-domain memory availability/visibility for `cmd`, which is a separate
    // submission on the same queue. Without this, the BLAS build below reading aabb_buffer as an
    // acceleration-structure build input is a Read-After-Write hazard per the Vulkan memory model.
    VkMemoryBarrier emit_to_build{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &emit_to_build, 0, nullptr, 0, nullptr);

    recordBlasBuildChunks(*this, cmd, aabb_buffer.address, /*update=*/false, /*override_prims=*/occupied);
    accel_ever_built = true;
    frames_since_full_rebuild = 0;
    stat_full_rebuilds++;
    slot_build_mode[frame_idx] = BlasBuild::Rebuild;

    VkMemoryBarrier blas_to_tlas{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &blas_to_tlas, 0, nullptr, 0, nullptr);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 2);
    }

    recordTlasBuild(*this, cmd, scene_tlas, tlas_scratch, tlas_instance_buffer.address,
        static_cast<uint32_t>(scene_blas.size()));

    VkMemoryBarrier to_trace{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &to_trace, 0, nullptr, 0, nullptr);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 3);
    }

    // Log the aggregation result once, on the first build for this scene (frame_idx is the in-flight
    // slot index 0..FRAMES-1, not a build counter, so gating on it would reprint every FRAMES frames).
    if (first_build)
    {
        spdlog::info("Path tracing: LOD emitted {} occupied cells (reduction {:.0f}:1 vs {} particles)",
            occupied, occupied ? double(particle_count) / double(occupied) : 0.0, particle_count);
    }
}

void RayTracingContext::destroyFrameResources()
{
    if (composite_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, composite_pipeline, nullptr);
        composite_pipeline = VK_NULL_HANDLE;
    }
    for (auto& si : storage_images)
    {
        if (si.view != VK_NULL_HANDLE)   { vkDestroyImageView(device, si.view, nullptr); }
        if (si.image != VK_NULL_HANDLE)  { vkDestroyImage(device, si.image, nullptr); }
        if (si.memory != VK_NULL_HANDLE) { vkFreeMemory(device, si.memory, nullptr); }
    }
    storage_images.clear();

    if (accum_image.view != VK_NULL_HANDLE)   { vkDestroyImageView(device, accum_image.view, nullptr); }
    if (accum_image.image != VK_NULL_HANDLE)  { vkDestroyImage(device, accum_image.image, nullptr); }
    if (accum_image.memory != VK_NULL_HANDLE) { vkFreeMemory(device, accum_image.memory, nullptr); }
    accum_image = {};

    auto destroy_images = [&](std::vector<StorageImage>& imgs) {
        for (auto& si : imgs)
        {
            if (si.view != VK_NULL_HANDLE)   { vkDestroyImageView(device, si.view, nullptr); }
            if (si.image != VK_NULL_HANDLE)  { vkDestroyImage(device, si.image, nullptr); }
            if (si.memory != VK_NULL_HANDLE) { vkFreeMemory(device, si.memory, nullptr); }
        }
        imgs.clear();
    };
    destroy_images(gbuffer_images);
    destroy_images(denoise_ping[0]);
    destroy_images(denoise_ping[1]);
}

void RayTracingContext::recordTrace(VkCommandBuffer cmd, uint32_t frame_idx,
    const RtPushConstants& pc, bool leave_image_general)
{
    // Capture the backdrop (background * intensity) for recordDenoise's final tone-map pass, which
    // shows it on primary-miss pixels just like the non-denoised raygen path.
    backdrop_color = glm::vec3(pc.sky_color) * pc.sky_color.w;

    auto image = storage_images[frame_idx].image;

    // Transition the storage image to GENERAL for raygen writes (contents discarded).
    VkImageMemoryBarrier to_general{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &to_general);

    // The accumulator persists across frames and is read+written by the raygen every frame. With
    // frames in flight this creates a cross-frame RAW/WAW hazard on a single shared image, so serialize
    // each frame's accumulator access after the previous frame's (same queue, submission-ordered).
    VkImageMemoryBarrier accum_barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = accum_image.image,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &accum_barrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
        rt_pipeline_layout, 0, 1, &rt_sets[frame_idx], 0, nullptr);
    // Inject the AABB buffer's device address so the sphere intersection shader can read each
    // primitive's bounds; the engine fills the rest of the push constants (camera, sun, etc.).
    RtPushConstants push = pc;
    push.aabbs = aabb_buffer.address;
    // Pack the BLAS chunk size into the unused cam_pos.w lane (as uint bits): the intersection shader
    // uses it to turn a chunk-local PrimitiveIndex into a global particle index.
    std::memcpy(&push.cam_pos.w, &blas_chunk_prims, sizeof(uint32_t));
    vkCmdPushConstants(cmd, rt_pipeline_layout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
        | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
        0, sizeof(RtPushConstants), &push);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 4);
    }
    api.cmdTraceRays(cmd, &raygen_region, &miss_region, &hit_region, &callable_region,
        extent.width, extent.height, 1);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 5);
    }

    // Transition to SHADER_READ_ONLY so the composite fragment shader can sample it -- unless the
    // denoiser will run next, in which case leave it in GENERAL for the à-trous final write.
    if (!leave_image_general)
    {
        VkImageMemoryBarrier to_read{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_read);
    }
}

void RayTracingContext::recordDenoise(VkCommandBuffer cmd, uint32_t frame_idx)
{
    if (gbuffer_images.empty() || atrous_pipeline == VK_NULL_HANDLE) { return; }

    // The raygen just wrote the accumulator and the G-buffer (RT shader stage). Make those visible
    // to the à-trous compute reads. The display image is still GENERAL (recordTrace was asked to
    // leave it so); the final pass writes it here.
    VkImageMemoryBarrier to_compute[2] = {
        { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = accum_image.image, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
        { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = gbuffer_images[frame_idx].image, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, to_compute);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atrous_pipeline);
    uint32_t gx = (extent.width + 7) / 8, gy = (extent.height + 7) / 8;

    // Edge-stopping falloffs. Colour is permissive (HDR radiance is noisy); normal/depth are tight
    // so silhouettes and depth discontinuities stop the blur. Constant across passes here.
    for (uint32_t pass = 0; pass < ATROUS_PASSES; ++pass)
    {
        AtrousPush push{
            .dims_x = static_cast<int32_t>(extent.width),
            .dims_y = static_cast<int32_t>(extent.height),
            .step_width = 1 << pass, // 1, 2, 4
            .c_phi = 4.0f, .n_phi = 0.1f, .p_phi = 0.5f,
            .tonemap = (pass + 1u == ATROUS_PASSES) ? 1u : 0u, // final pass tonemaps to the display
            .sky_r = backdrop_color.r, .sky_g = backdrop_color.g, .sky_b = backdrop_color.b,
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atrous_pipeline_layout,
            0, 1, &atrous_sets[frame_idx * ATROUS_PASSES + pass], 0, nullptr);
        vkCmdPushConstants(cmd, atrous_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(push), &push);
        vkCmdDispatch(cmd, gx, gy, 1);

        // Serialize this pass's writes before the next pass reads the same ping buffer (or, for the
        // final pass, before the composite samples the display image).
        bool last = (pass + 1u == ATROUS_PASSES);
        VkImageMemoryBarrier b{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = last ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = last ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = last ? storage_images[frame_idx].image
                          : denoise_ping[pass % 2][frame_idx].image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            last ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
    }
}

void RayTracingContext::readTimings(uint32_t frame_idx)
{
    if (timing_pool == VK_NULL_HANDLE) { return; }
    uint64_t ts[TS_PER_FRAME] = {};
    // No WAIT_BIT: the caller only reads after the frame's fence, so results are ready; if for any
    // reason they are not (VK_NOT_READY), keep the previous values rather than stalling.
    VkResult r = vkGetQueryPoolResults(device, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME,
        sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) { return; }
    // Slots: 0 build start, 1 after AABB, 2 after BLAS, 3 after TLAS (build end), 4/5 trace.
    double ms_per_tick = static_cast<double>(timestamp_period) / 1.0e6;
    last_aabb_ms  = static_cast<double>(ts[1] - ts[0]) * ms_per_tick;
    last_blas_ms  = static_cast<double>(ts[2] - ts[1]) * ms_per_tick;
    last_tlas_ms  = static_cast<double>(ts[3] - ts[2]) * ms_per_tick;
    last_build_ms = static_cast<double>(ts[3] - ts[0]) * ms_per_tick;
    last_trace_ms = static_cast<double>(ts[5] - ts[4]) * ms_per_tick;
    have_timings = true; // readings are now real, not the session-start defaults
    // Pair the build time with the mode that produced it. slot_build_mode[frame_idx] was stamped by
    // recordUpdateScene FRAMES frames ago -- the same frame these timestamps are from -- so the split
    // and its refit/rebuild/skip label describe one frame, not two FRAMES apart.
    last_build_mode = slot_build_mode[frame_idx];
}

void RayTracingContext::recordComposite(VkCommandBuffer cmd, uint32_t frame_idx)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, composite_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        composite_pipeline_layout, 0, 1, &composite_sets[frame_idx], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void RayTracingContext::destroy()
{
    destroyFrameResources();
    if (timing_pool) { vkDestroyQueryPool(device, timing_pool, nullptr); }
    if (composite_pipeline_layout) { vkDestroyPipelineLayout(device, composite_pipeline_layout, nullptr); }
    if (composite_set_layout)      { vkDestroyDescriptorSetLayout(device, composite_set_layout, nullptr); }
    if (composite_pool)            { vkDestroyDescriptorPool(device, composite_pool, nullptr); }
    if (composite_sampler)         { vkDestroySampler(device, composite_sampler, nullptr); }

    if (atrous_pipeline)        { vkDestroyPipeline(device, atrous_pipeline, nullptr); }
    if (atrous_pipeline_layout) { vkDestroyPipelineLayout(device, atrous_pipeline_layout, nullptr); }
    if (atrous_set_layout)      { vkDestroyDescriptorSetLayout(device, atrous_set_layout, nullptr); }
    if (atrous_pool)            { vkDestroyDescriptorPool(device, atrous_pool, nullptr); }

    if (rt_pipeline)        { vkDestroyPipeline(device, rt_pipeline, nullptr); }
    if (rt_pipeline_layout) { vkDestroyPipelineLayout(device, rt_pipeline_layout, nullptr); }
    if (rt_set_layout)      { vkDestroyDescriptorSetLayout(device, rt_set_layout, nullptr); }
    if (rt_pool)            { vkDestroyDescriptorPool(device, rt_pool, nullptr); }
    destroyBuffer(device, sbt_buffer);

    // AABB-writer compute (push-constant-only: no descriptor set/pool)
    if (iw_pipeline)        { vkDestroyPipeline(device, iw_pipeline, nullptr); }
    if (iw_pipeline_layout) { vkDestroyPipelineLayout(device, iw_pipeline_layout, nullptr); }

    // LOD grid-aggregation compute (scatter + emit)
    if (lod_scatter_pipeline   != VK_NULL_HANDLE) { vkDestroyPipeline(device, lod_scatter_pipeline, nullptr); }
    if (lod_emit_pipeline      != VK_NULL_HANDLE) { vkDestroyPipeline(device, lod_emit_pipeline, nullptr); }
    if (lod_scatter_layout     != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, lod_scatter_layout, nullptr); }
    if (lod_emit_layout        != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, lod_emit_layout, nullptr); }
    if (lod_scatter_set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, lod_scatter_set_layout, nullptr); }
    if (lod_emit_set_layout    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, lod_emit_set_layout, nullptr); }
    if (lod_desc_pool          != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device, lod_desc_pool, nullptr); }

    // Shared dynamic scene (one-instance TLAS + AABB BLAS + AABB buffer + scratch)
    if (scene_tlas.handle)
    {
        api.destroyAccelerationStructure(device, scene_tlas.handle, nullptr);
    }
    destroyBuffer(device, scene_tlas.buffer);
    destroyBuffer(device, tlas_instance_buffer);
    destroyBuffer(device, tlas_scratch);

    for (auto& blas : scene_blas)
    {
        if (blas.handle) { api.destroyAccelerationStructure(device, blas.handle, nullptr); }
        destroyBuffer(device, blas.buffer);
    }
    scene_blas.clear();
    destroyBuffer(device, blas_scratch);
    destroyBuffer(device, aabb_buffer);

    // LOD grid-aggregation buffers (allocated in bindScene only when lod_cells > 0)
    destroyBuffer(device, lod_cellcount_buffer);
    destroyBuffer(device, lod_counter_buffer);
}

} // namespace mimir
