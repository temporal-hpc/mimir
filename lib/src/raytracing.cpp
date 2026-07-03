#include "mimir/raytracing.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>     // std::max
#include <cstring>       // std::memcpy
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

// ---- Icosphere geometry ----------------------------------------------------------

struct Mesh
{
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};

Mesh makeIcosphere(uint32_t subdiv)
{
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<glm::vec3> verts = {
        {-1, t, 0}, {1, t, 0}, {-1,-t, 0}, {1,-t, 0},
        {0,-1, t}, {0, 1, t}, {0,-1,-t}, {0, 1,-t},
        {t, 0,-1}, {t, 0, 1}, {-t, 0,-1}, {-t, 0, 1},
    };
    for (auto& v : verts) { v = glm::normalize(v); }

    std::vector<glm::uvec3> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    };

    // Midpoint subdivision with an edge cache so shared edges reuse vertices.
    for (uint32_t s = 0; s < subdiv; ++s)
    {
        std::unordered_map<uint64_t, uint32_t> cache;
        auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
            uint64_t key = a < b ? (uint64_t(a) << 32 | b) : (uint64_t(b) << 32 | a);
            auto it = cache.find(key);
            if (it != cache.end()) { return it->second; }
            auto m = glm::normalize((verts[a] + verts[b]) * 0.5f);
            uint32_t idx = static_cast<uint32_t>(verts.size());
            verts.push_back(m);
            cache.emplace(key, idx);
            return idx;
        };
        std::vector<glm::uvec3> next;
        next.reserve(faces.size() * 4);
        for (const auto& f : faces)
        {
            uint32_t a = midpoint(f.x, f.y);
            uint32_t b = midpoint(f.y, f.z);
            uint32_t c = midpoint(f.z, f.x);
            next.push_back({f.x, a, c});
            next.push_back({f.y, b, a});
            next.push_back({f.z, c, b});
            next.push_back({a, b, c});
        }
        faces.swap(next);
    }

    Mesh mesh;
    mesh.positions = std::move(verts);
    mesh.indices.reserve(faces.size() * 3);
    for (const auto& f : faces) { mesh.indices.insert(mesh.indices.end(), {f.x, f.y, f.z}); }
    return mesh;
}

// ---- Acceleration-structure build helpers ----------------------------------------

// Allocates the AS backing buffer, creates the acceleration structure, builds it with a
// temporary scratch buffer via a one-time submit, and resolves its device address.
AccelStruct buildAccelStruct(RayTracingContext& ctx,
    VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureGeometryKHR& geometry,
    uint32_t primitive_count, VkBuildAccelerationStructureFlagsKHR flags)
{
    VkAccelerationStructureBuildGeometryInfoKHR build_info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type  = type,
        .flags = flags,
        .mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries   = &geometry,
        .ppGeometries  = nullptr,
        .scratchData   = {},
    };

    VkAccelerationStructureBuildSizesInfoKHR sizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
        .pNext = nullptr,
        .accelerationStructureSize = 0,
        .updateScratchSize = 0,
        .buildScratchSize = 0,
    };
    ctx.api.getBuildSizes(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &primitive_count, &sizes);

    AccelStruct as{};
    as.buffer = makeBuffer(ctx, sizes.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, DEVICE_LOCAL, true);

    VkAccelerationStructureCreateInfoKHR create_info{
        .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .pNext  = nullptr,
        .createFlags = 0,
        .buffer = as.buffer.buffer,
        .offset = 0,
        .size   = sizes.accelerationStructureSize,
        .type   = type,
        .deviceAddress = 0,
    };
    validation::checkVulkan(ctx.api.createAccelerationStructure(
        ctx.device, &create_info, nullptr, &as.handle));
    build_info.dstAccelerationStructure = as.handle;

    // Scratch buffer, aligned to the device's minimum scratch offset alignment.
    auto scratch_align = ctx.accel_props.minAccelerationStructureScratchOffsetAlignment;
    auto scratch = makeBuffer(ctx, sizes.buildScratchSize + scratch_align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, DEVICE_LOCAL, true);
    build_info.scratchData.deviceAddress = alignUp(scratch.address, scratch_align);

    VkAccelerationStructureBuildRangeInfoKHR range{
        .primitiveCount = primitive_count,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
    const VkAccelerationStructureBuildRangeInfoKHR* p_range = &range;
    ctx.submit([&](VkCommandBuffer cmd) {
        ctx.api.cmdBuildAccelerationStructures(cmd, 1, &build_info, &p_range);
    });
    destroyBuffer(ctx.device, scratch);

    VkAccelerationStructureDeviceAddressInfoKHR addr_info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .pNext = nullptr,
        .accelerationStructure = as.handle,
    };
    as.address = ctx.api.getAccelStructAddress(ctx.device, &addr_info);
    return as;
}

void buildIcosphereBlas(RayTracingContext& ctx, uint32_t subdiv)
{
    auto mesh = makeIcosphere(subdiv);
    ctx.vertex_count = static_cast<uint32_t>(mesh.positions.size());
    ctx.index_count  = static_cast<uint32_t>(mesh.indices.size());

    VkDeviceSize vsize = ctx.vertex_count * sizeof(glm::vec3);
    VkDeviceSize isize = ctx.index_count * sizeof(uint32_t);
    auto build_usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    ctx.vertex_buffer = makeBuffer(ctx, vsize, build_usage, HOST_VISIBLE, true);
    ctx.index_buffer  = makeBuffer(ctx, isize, build_usage, HOST_VISIBLE, true);
    uploadBuffer(ctx.device, ctx.vertex_buffer, mesh.positions.data(), vsize);
    uploadBuffer(ctx.device, ctx.index_buffer, mesh.indices.data(), isize);

    VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry = {},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    geometry.geometry.triangles = VkAccelerationStructureGeometryTrianglesDataKHR{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .pNext = nullptr,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
        .vertexData   = { .deviceAddress = ctx.vertex_buffer.address },
        .vertexStride = sizeof(glm::vec3),
        .maxVertex    = ctx.vertex_count - 1,
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = { .deviceAddress = ctx.index_buffer.address },
        .transformData = { .deviceAddress = 0 },
    };

    ctx.blas = buildAccelStruct(ctx, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        geometry, ctx.index_count / 3,
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
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
    // Descriptor set layout: TLAS + display storage image + persistent accumulator image.
    VkDescriptorSetLayoutBinding bindings[3] = {
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
    };
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .bindingCount = 3, .pBindings = bindings,
    };
    validation::checkVulkan(vkCreateDescriptorSetLayout(
        ctx.device, &set_info, nullptr, &ctx.rt_set_layout));

    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                    | VK_SHADER_STAGE_MISS_BIT_KHR,
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
        .entrypoints = { "raygenMain", "missMain", "closestHitMain" },
        .specializations = {},
    };
    auto stages = builder.compileModule(ctx.device, params);
    std::filesystem::current_path(orig_path);

    if (stages.size() != 3)
    {
        spdlog::error("pathtrace.slang: expected 3 RT stages, got {}", stages.size());
    }

    // Group each stage: raygen (general), miss (general), closest-hit (triangles hit group).
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
             hit_idx = VK_SHADER_UNUSED_KHR;
    for (uint32_t i = 0; i < stages.size(); ++i)
    {
        switch (stages[i].stage)
        {
            case VK_SHADER_STAGE_RAYGEN_BIT_KHR:      raygen_idx = i; break;
            case VK_SHADER_STAGE_MISS_BIT_KHR:        miss_idx = i;   break;
            case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR: hit_idx = i;    break;
            default: break;
        }
    }
    groups.push_back(general_group(raygen_idx));
    groups.push_back(general_group(miss_idx));
    groups.push_back(VkRayTracingShaderGroupCreateInfoKHR{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
        .pNext = nullptr,
        .type  = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
        .generalShader = VK_SHADER_UNUSED_KHR,
        .closestHitShader = hit_idx,
        .anyHitShader = VK_SHADER_UNUSED_KHR,
        .intersectionShader = VK_SHADER_UNUSED_KHR,
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

    // Descriptor pool sized for a few frames in flight (TLAS + two storage images per set:
    // the display image and the shared accumulator).
    VkDescriptorPoolSize rt_pool_sizes[2] = {
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16 },
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

// Push constants for pathtrace_instances.slang (20 bytes): count, radius, BLAS ref lo/hi,
// and the number of materials instances cycle through (writer sets sbtRecordOffset = idx % this).
struct InstanceWriterPush
{
    uint32_t count;
    float radius;
    uint32_t blas_ref_lo;
    uint32_t blas_ref_hi;
    uint32_t material_count;
};

void createInstanceWriter(RayTracingContext& ctx)
{
    // Two storage buffers: positions (read) + instances (write).
    VkDescriptorSetLayoutBinding bindings[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = nullptr },
    };
    VkDescriptorSetLayoutCreateInfo set_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .bindingCount = 2, .pBindings = bindings,
    };
    validation::checkVulkan(vkCreateDescriptorSetLayout(
        ctx.device, &set_info, nullptr, &ctx.iw_set_layout));

    VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(InstanceWriterPush),
    };
    VkPipelineLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr, .flags = 0,
        .setLayoutCount = 1, .pSetLayouts = &ctx.iw_set_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range,
    };
    validation::checkVulkan(vkCreatePipelineLayout(
        ctx.device, &layout_info, nullptr, &ctx.iw_pipeline_layout));

    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(getDefaultShaderPath());
    auto builder = ShaderBuilder::make();
    ShaderCompileParams params{
        .module_path = "shaders/pathtrace_instances.slang",
        .entrypoints = { "writeInstancesMain" }, .specializations = {},
    };
    auto stages = builder.compileModule(ctx.device, params);
    std::filesystem::current_path(orig_path);
    if (stages.size() != 1)
    {
        spdlog::error("pathtrace_instances.slang: expected 1 compute stage, got {}", stages.size());
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

    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * RayTracingContext::FRAMES };
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .maxSets = RayTracingContext::FRAMES,
        .poolSizeCount = 1, .pPoolSizes = &pool_size,
    };
    validation::checkVulkan(vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &ctx.iw_pool));
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
    alloc(ctx.iw_pool, ctx.iw_set_layout, ctx.iw_sets);
}

} // namespace

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

    // GPU-timestamp timing pool for the HUD/CSV (FRAMES*4 timestamps; see the header). Disabled
    // if the device reports a 0 timestamp period (no timestamp support), leaving timings at 0.
    ctx.timestamp_period = props2.properties.limits.timestampPeriod;
    if (ctx.timestamp_period > 0.f)
    {
        VkQueryPoolCreateInfo qinfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = FRAMES * 4, .pipelineStatistics = 0,
        };
        validation::checkVulkan(vkCreateQueryPool(device, &qinfo, nullptr, &ctx.timing_pool));
        vkResetQueryPool(device, ctx.timing_pool, 0, FRAMES * 4);
    }

    buildIcosphereBlas(ctx, subdiv);
    createRtPipeline(ctx);
    createCompositeResources(ctx);
    createInstanceWriter(ctx);
    allocateDescriptorSets(ctx);

    spdlog::info("Path tracing ready: icosphere subdiv {} ({} tris); scene bound per view",
        subdiv, ctx.index_count / 3);
    return ctx;
}

void RayTracingContext::createFrameResources(VkExtent2D new_extent, VkRenderPass render_pass)
{
    extent = new_extent;

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
        VkDescriptorImageInfo sampled_info{
            .sampler = composite_sampler, .imageView = storage_images[i].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        VkWriteDescriptorSet writes[3] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = rt_sets[i], .dstBinding = 1, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &storage_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = rt_sets[i], .dstBinding = 2, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &accum_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = composite_sets[i], .dstBinding = 0, .dstArrayElement = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &sampled_info, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
        };
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
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

void RayTracingContext::bindScene(VkBuffer positions, uint32_t count, float radius, glm::vec4 color)
{
    position_buffer = positions;
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

    // Per-frame instance buffer (written by the compute writer, read by the TLAS build) and
    // per-frame TLAS + scratch. Instance buffer needs STORAGE (compute write),
    // AS_BUILD_INPUT (TLAS read), and a device address (TLAS build input).
    VkDeviceSize inst_size = VkDeviceSize(count) * sizeof(VkAccelerationStructureInstanceKHR);
    for (uint32_t f = 0; f < FRAMES; ++f)
    {
        instance_buffers[f] = makeBuffer(*this, inst_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            DEVICE_LOCAL, true);
        createDynamicTlas(*this, scene_tlas[f], tlas_scratch[f],
            instance_buffers[f].address, count);

        // Instance-writer descriptors: positions (shared) + this frame's instance buffer.
        VkDescriptorBufferInfo pos_info{ position_buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo inst_info{ instance_buffers[f].buffer, 0, VK_WHOLE_SIZE };
        // RT descriptor: this frame's TLAS (binding 0).
        VkWriteDescriptorSetAccelerationStructureKHR as_write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr, .accelerationStructureCount = 1,
            .pAccelerationStructures = &scene_tlas[f].handle,
        };
        VkWriteDescriptorSet writes[3] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = iw_sets[f], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pImageInfo = nullptr, .pBufferInfo = &pos_info, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr,
              .dstSet = iw_sets[f], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pImageInfo = nullptr, .pBufferInfo = &inst_info, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = &as_write,
              .dstSet = rt_sets[f], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
              .pImageInfo = nullptr, .pBufferInfo = nullptr, .pTexelBufferView = nullptr },
        };
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }

    // Populate instances and build each frame's initial TLAS (positions are already valid at
    // bind time), so the first rendered frame has a coherent AS even before its own update.
    submit([&](VkCommandBuffer cmd) {
        for (uint32_t f = 0; f < FRAMES; ++f)
        {
            recordUpdateScene(cmd, f);
        }
    });

    scene_bound = true;
    spdlog::info("Path tracing scene bound: {} instances, radius {:.4f}", count, radius);
}

void RayTracingContext::recordUpdateScene(VkCommandBuffer cmd, uint32_t frame_idx)
{
    if (particle_count == 0) { return; }

    // Reset this frame's 4 timestamps (previous readings for frame_idx were already read back in
    // renderFrame) and stamp the start of the TLAS build.
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd, timing_pool, frame_idx * 4, 4);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * 4 + 0);
    }

    // 1) Instance writer: fill this frame's instance buffer from the live positions.
    InstanceWriterPush push{
        .count = particle_count, .radius = particle_radius,
        .blas_ref_lo = static_cast<uint32_t>(blas.address & 0xFFFFFFFFu),
        .blas_ref_hi = static_cast<uint32_t>(blas.address >> 32),
        .material_count = std::max(instance_material_count, 1u),
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, iw_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, iw_pipeline_layout,
        0, 1, &iw_sets[frame_idx], 0, nullptr);
    vkCmdPushConstants(cmd, iw_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), &push);
    vkCmdDispatch(cmd, (particle_count + 63) / 64, 1, 1);

    // Barrier: compute writes -> AS build reads the instance buffer.
    VkMemoryBarrier to_build{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &to_build, 0, nullptr, 0, nullptr);

    // 2) Rebuild this frame's TLAS from the updated instances.
    recordTlasBuild(*this, cmd, scene_tlas[frame_idx], tlas_scratch[frame_idx],
        instance_buffers[frame_idx].address, particle_count);

    // Barrier: AS build writes -> ray tracing reads the TLAS.
    VkMemoryBarrier to_trace{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &to_trace, 0, nullptr, 0, nullptr);

    // Stamp the end of the TLAS build (instance writer + build barriers).
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * 4 + 1);
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
}

void RayTracingContext::recordTrace(VkCommandBuffer cmd, uint32_t frame_idx,
    const RtPushConstants& pc)
{
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
    vkCmdPushConstants(cmd, rt_pipeline_layout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
        | VK_SHADER_STAGE_MISS_BIT_KHR, 0, sizeof(RtPushConstants), &pc);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * 4 + 2);
    }
    api.cmdTraceRays(cmd, &raygen_region, &miss_region, &hit_region, &callable_region,
        extent.width, extent.height, 1);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * 4 + 3);
    }

    // Transition to SHADER_READ_ONLY so the composite fragment shader can sample it.
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

void RayTracingContext::readTimings(uint32_t frame_idx)
{
    if (timing_pool == VK_NULL_HANDLE) { return; }
    uint64_t ts[4] = { 0, 0, 0, 0 };
    // No WAIT_BIT: the caller only reads after the frame's fence, so results are ready; if for any
    // reason they are not (VK_NOT_READY), keep the previous values rather than stalling.
    VkResult r = vkGetQueryPoolResults(device, timing_pool, frame_idx * 4, 4,
        sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) { return; }
    double ms_per_tick = static_cast<double>(timestamp_period) / 1.0e6;
    last_tlas_ms  = static_cast<double>(ts[1] - ts[0]) * ms_per_tick;
    last_trace_ms = static_cast<double>(ts[3] - ts[2]) * ms_per_tick;
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

    if (rt_pipeline)        { vkDestroyPipeline(device, rt_pipeline, nullptr); }
    if (rt_pipeline_layout) { vkDestroyPipelineLayout(device, rt_pipeline_layout, nullptr); }
    if (rt_set_layout)      { vkDestroyDescriptorSetLayout(device, rt_set_layout, nullptr); }
    if (rt_pool)            { vkDestroyDescriptorPool(device, rt_pool, nullptr); }
    destroyBuffer(device, sbt_buffer);

    // Instance-writer compute
    if (iw_pipeline)        { vkDestroyPipeline(device, iw_pipeline, nullptr); }
    if (iw_pipeline_layout) { vkDestroyPipelineLayout(device, iw_pipeline_layout, nullptr); }
    if (iw_set_layout)      { vkDestroyDescriptorSetLayout(device, iw_set_layout, nullptr); }
    if (iw_pool)            { vkDestroyDescriptorPool(device, iw_pool, nullptr); }

    // Per-frame dynamic scene (TLAS + instances + scratch)
    for (uint32_t f = 0; f < FRAMES; ++f)
    {
        if (scene_tlas[f].handle)
        {
            api.destroyAccelerationStructure(device, scene_tlas[f].handle, nullptr);
        }
        destroyBuffer(device, scene_tlas[f].buffer);
        destroyBuffer(device, instance_buffers[f]);
        destroyBuffer(device, tlas_scratch[f]);
    }

    if (blas.handle) { api.destroyAccelerationStructure(device, blas.handle, nullptr); }
    destroyBuffer(device, blas.buffer);
    destroyBuffer(device, vertex_buffer);
    destroyBuffer(device, index_buffer);
}

} // namespace mimir
