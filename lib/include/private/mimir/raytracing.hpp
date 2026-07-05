#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <functional> // std::function
#include <vector> // std::vector

namespace mimir
{

// KHR ray tracing / acceleration-structure device entry points. These are extension
// functions and must be resolved with vkGetDeviceProcAddr after device creation.
struct RayTracingApi
{
    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelStructAddress = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createRayTracingPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getShaderGroupHandles = nullptr;
    PFN_vkCmdTraceRaysKHR cmdTraceRays = nullptr;

    static RayTracingApi load(VkDevice device);
};

// A device buffer paired with its memory and (optionally) its device address.
struct RtBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
};

// A Vulkan acceleration structure plus its backing storage buffer.
struct AccelStruct
{
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    RtBuffer buffer;
    VkDeviceAddress address = 0;
};

// Push constants shared by the raygen/closest-hit/miss shaders (must match pathtrace.slang).
// The camera is passed as an explicit world-space basis (extracted from the LookAt
// camera-to-world matrix) so the raygen does a plain pinhole projection, avoiding mimir's
// pre-transposed glm/slang matrix convention. 120 bytes, within the 128-byte guaranteed limit.
struct RtPushConstants
{
    glm::vec4 cam_pos;     // xyz camera world position; w unused
    glm::vec4 cam_right;   // xyz world-space camera basis; w = light_color R (PT sun tint)
    glm::vec4 cam_up;      // xyz ...                       w = light_color G
    glm::vec4 cam_forward; // xyz ...                       w = light_color B
    glm::vec4 sun_dir;     // xyz world-space direction TO the sun (normalized); w unused
    glm::vec4 sky_color;   // environment/background color; w = intensity
    float tan_half_fov = 0.f; // tan(vertical_fov / 2)
    float aspect = 1.f;       // width / height
    uint32_t frame_index = 0;
    uint32_t spp = 1;
    uint32_t bounces = 4;
    // Temporal accumulation: number of frames already averaged into the accumulator image (0 on
    // the first frame after a reset). The engine resets to 0 on camera motion or a new simulation
    // iteration, then increments; the raygen keeps a running HDR mean so static views converge.
    uint32_t accum_frame = 0;
};

// Per-material surface parameters, stored as shader-record data in each SBT hit record (must match
// MaterialData in pathtrace.slang: float3 albedo + float emission, 16 bytes). The SBT holds one hit
// record per registered material; an instance's shaderBindingTableRecordOffset picks which record
// (material) it shades with, so a single closest-hit shader serves many materials. This is the
// multi-material SBT the library exposes; the particles sample registers one material, but the
// machinery supports several (e.g. emissive area lights alongside diffuse surfaces).
struct MaterialData
{
    glm::vec3 albedo{0.82f, 0.82f, 0.88f}; // diffuse reflectance
    float emission = 0.f;                  // 0 = lambertian; >0 emits albedo*emission
};

// Path-tracing render context (LightModel::PathTracing). Owns the icosphere BLAS, the
// scene TLAS, the ray-tracing pipeline + SBT, and the per-frame storage images plus the
// fullscreen composite pipeline that samples them into the raster render pass (so the
// existing ImGui HUD/present machinery is untouched). RT pipeline/SBT/BLAS/TLAS are
// resolution-independent (context lifetime); storage images + composite pipeline depend
// on the swapchain extent (frame-resource lifetime, rebuilt on resize).
struct RayTracingContext
{
    using SubmitFn = std::function<void(std::function<void(VkCommandBuffer)>)>;

    // Frames in flight for per-frame PT resources (storage image, TLAS, instance buffer).
    // Must match the engine's MAX_FRAMES_IN_FLIGHT; frame indices are render_timeline % this.
    static constexpr uint32_t FRAMES = 3;

    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    RayTracingApi api{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accel_props{};

    // One-time GPU submit callback (engine's immediateSubmit) used for AS builds/transitions.
    SubmitFn submit;

    // Icosphere geometry + bottom-level acceleration structure (one, built once)
    RtBuffer vertex_buffer;
    RtBuffer index_buffer;
    uint32_t index_count = 0;
    uint32_t vertex_count = 0;
    AccelStruct blas;

    // Dynamic scene (Phase 2): a per-frame TLAS rebuilt each frame from the live interop
    // position buffer. Bound after view creation via bindScene(). Per-frame (indexed by
    // frame-in-flight) so a frame's TLAS/instances are never overwritten while still in use.
    bool scene_bound = false;
    VkBuffer position_buffer = VK_NULL_HANDLE; // interop positions (owned by the view, not us)
    uint32_t particle_count = 0;
    float particle_radius = 0.f;
    glm::vec4 particle_color{0.82f, 0.82f, 0.88f, 1.f}; // surface albedo (from the view's color)
    AccelStruct scene_tlas[FRAMES];       // per-frame TLAS
    RtBuffer instance_buffers[FRAMES];    // per-frame VkAccelerationStructureInstanceKHR[]
    RtBuffer tlas_scratch[FRAMES];        // persistent per-frame build scratch

    // Instance-writer compute (fills instance_buffers[frame] from position_buffer)
    VkDescriptorSetLayout iw_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout iw_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline iw_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool iw_pool = VK_NULL_HANDLE;
    VkDescriptorSet iw_sets[FRAMES] = {};

    // Ray-tracing pipeline + shader binding table
    VkDescriptorSetLayout rt_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout rt_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline rt_pipeline = VK_NULL_HANDLE;
    RtBuffer sbt_buffer;
    VkStridedDeviceAddressRegionKHR raygen_region{};
    VkStridedDeviceAddressRegionKHR miss_region{};
    VkStridedDeviceAddressRegionKHR hit_region{};
    VkStridedDeviceAddressRegionKHR callable_region{};

    // Multi-material shader binding table. The hit region holds one record per registered material
    // (each = the closest-hit group handle followed by that material's MaterialData shader record).
    // An instance's shaderBindingTableRecordOffset selects its record, so many materials share one
    // closest-hit shader. The particles sample keeps instance_material_count == 1 (every instance
    // uses material 0), but the SBT is built for all `materials`, so the capability is always live.
    std::vector<MaterialData> materials;
    uint32_t instance_material_count = 1; // # of materials instances cycle through (writer: idx % this)
    uint32_t shader_handle_size = 0;      // rt_props.shaderGroupHandleSize (record handle prefix size)
    VkDeviceSize sbt_hit_records_offset = 0; // byte offset of the first hit record within sbt_buffer

    // Descriptor pool + per-frame RT sets (binding 0 = TLAS, binding 1 = storage image)
    VkDescriptorPool rt_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> rt_sets;

    // Fullscreen composite (samples storage image into the raster color attachment)
    VkSampler composite_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout composite_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout composite_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline composite_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool composite_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> composite_sets;

    // Per-frame storage images (RGBA16F), extent-dependent (frame-resource lifetime)
    struct StorageImage
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    std::vector<StorageImage> storage_images;
    // Persistent HDR temporal-accumulation image (RGBA32F), one shared across frames in flight:
    // the raygen keeps a running mean of the pre-tonemap radiance here so a static view converges.
    // Extent-dependent (rebuilt on resize, which also restarts accumulation). Bound at RT binding 2.
    StorageImage accum_image;
    // Denoiser resources (extent-dependent). gbuffer_images hold the first-hit guide (RT binding 3:
    // xyz world normal, w eye-space depth) the a-trous filter edge-stops on; denoise_ping[0/1] are
    // the ping-pong scratch buffers the wavelet passes bounce through before the final tonemapped
    // write lands in the display image. Per-frame (fence-gated reuse), mirroring storage_images.
    std::vector<StorageImage> gbuffer_images;
    std::vector<StorageImage> denoise_ping[2];
    VkExtent2D extent{};
    static constexpr VkFormat storage_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr VkFormat accum_format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    static constexpr VkFormat gbuffer_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    static constexpr uint32_t ATROUS_PASSES  = 3; // wavelet iterations (step widths 1, 2, 4)
    uint32_t max_recursion = 2;

    // À-trous denoiser compute pipeline (resolution-independent) + its per-pass descriptor sets.
    // atrous_sets is indexed [frame * ATROUS_PASSES + pass]; each set wires (in_color, gbuffer,
    // out_color) for one wavelet pass, chaining accum -> ping0 -> ping1 -> display image.
    VkDescriptorSetLayout atrous_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout atrous_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline atrous_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool atrous_pool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> atrous_sets;

    // GPU-timestamp timing for the HUD/CSV: a query pool with FRAMES*4 timestamps (per frame:
    // 0/1 bracket the instance-writer + TLAS build, 2/3 bracket the trace). Read back with one
    // frame of latency (after the frame's fence). last_*_ms hold the most recent readings.
    VkQueryPool timing_pool = VK_NULL_HANDLE;
    float timestamp_period = 0.f; // nanoseconds per tick (0 = timestamps unsupported, disabled)
    double last_tlas_ms = 0.0;    // instance-writer + TLAS build time, last completed frame
    double last_trace_ms = 0.0;   // vkCmdTraceRays time, last completed frame

    // Build the resolution-independent context: loads the RT API, queries properties,
    // builds the icosphere BLAS and a static grid TLAS, and creates the RT pipeline + SBT.
    // `materials` seeds the multi-material SBT (one hit record each); if empty a single default
    // diffuse material is used. bindScene overwrites material 0's albedo with the view color.
    static RayTracingContext make(VkDevice device, VkPhysicalDevice gpu,
        VkPhysicalDeviceMemoryProperties mem_props, SubmitFn submit,
        uint32_t subdiv, uint32_t max_recursion,
        std::vector<MaterialData> materials = {});

    // Rewrite material `index`'s shader-record data into the SBT hit record (host-visible buffer).
    // Setup-time only (no in-flight frames): used by bindScene to push the view color into material 0.
    void updateMaterial(uint32_t index);

    // (Re)build the extent-dependent frame resources: storage images + composite pipeline,
    // and (re)point the storage-image/composite descriptor bindings at the new images.
    void createFrameResources(VkExtent2D extent, VkRenderPass render_pass);
    // Destroy the extent-dependent frame resources (call on swapchain rebuild).
    void destroyFrameResources();

    // Bind the dynamic scene: the interop position buffer (VkBuffer, particle_count points of
    // tightly-packed float3) drives a per-frame TLAS of icosphere instances of the given world
    // radius. Allocates the per-frame instance buffers/TLAS/scratch, wires the instance-writer
    // and RT (TLAS) descriptors, and builds an initial TLAS. Call once after view creation.
    void bindScene(VkBuffer positions, uint32_t particle_count, float radius, glm::vec4 color);

    // Record the per-frame scene update for this frame: dispatch the instance-writer compute
    // over the live positions, then rebuild this frame's TLAS. Must be recorded OUTSIDE a
    // render pass, before recordTrace. No-op if no scene is bound.
    void recordUpdateScene(VkCommandBuffer cmd, uint32_t frame_idx);

    // Record the ray-trace for the given frame into its storage image (adds the layout
    // barriers around vkCmdTraceRaysKHR). Must be recorded OUTSIDE a render pass. When
    // leave_image_general is set (the denoiser will post-process the display image), the display
    // image is left in GENERAL instead of transitioned to SHADER_READ_ONLY for the composite.
    void recordTrace(VkCommandBuffer cmd, uint32_t frame_idx, const RtPushConstants& pc,
        bool leave_image_general = false);

    // Record the à-trous denoiser passes for this frame: filter the accumulator (guided by the
    // G-buffer) through the ping-pong buffers and write the tonemapped result into the display
    // image, leaving it SHADER_READ_ONLY for the composite. Must be recorded OUTSIDE a render pass,
    // after recordTrace(..., leave_image_general=true). No-op if denoiser resources are absent.
    void recordDenoise(VkCommandBuffer cmd, uint32_t frame_idx);

    // Read back the given frame's TLAS-build/trace timestamps into last_tlas_ms/last_trace_ms.
    // Call after the frame's fence has signalled (its previous submission is complete) and BEFORE
    // recordUpdateScene resets the queries for the new frame. No-op if timing is unsupported.
    void readTimings(uint32_t frame_idx);
    // Record the fullscreen composite that samples the frame's storage image. Must be
    // recorded INSIDE the raster render pass, before the ImGui HUD.
    void recordComposite(VkCommandBuffer cmd, uint32_t frame_idx);

    // Full teardown (context-lifetime objects). Frame resources must be destroyed first.
    void destroy();
};

} // namespace mimir
