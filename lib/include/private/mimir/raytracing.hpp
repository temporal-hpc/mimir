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
    glm::vec4 cam_pos;     // camera world position (w unused)
    glm::vec4 cam_right;   // world-space camera basis (w unused)
    glm::vec4 cam_up;
    glm::vec4 cam_forward;
    glm::vec4 sun_dir;     // world-space direction TO the sun (normalized); w unused
    glm::vec4 sky_color;   // environment/background color; w = intensity
    float tan_half_fov = 0.f; // tan(vertical_fov / 2)
    float aspect = 1.f;       // width / height
    uint32_t frame_index = 0;
    uint32_t spp = 1;
    uint32_t bounces = 4;
    float albedo_r = 0.82f; // particle surface color (--pcolor); packed into the former pad slots
    float albedo_g = 0.82f; // so the struct stays at the 128-byte guaranteed push-constant limit
    float albedo_b = 0.88f;
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
    VkExtent2D extent{};
    static constexpr VkFormat storage_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    uint32_t max_recursion = 2;

    // Build the resolution-independent context: loads the RT API, queries properties,
    // builds the icosphere BLAS and a static grid TLAS, and creates the RT pipeline + SBT.
    static RayTracingContext make(VkDevice device, VkPhysicalDevice gpu,
        VkPhysicalDeviceMemoryProperties mem_props, SubmitFn submit,
        uint32_t subdiv, uint32_t max_recursion);

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
    // barriers around vkCmdTraceRaysKHR). Must be recorded OUTSIDE a render pass.
    void recordTrace(VkCommandBuffer cmd, uint32_t frame_idx, const RtPushConstants& pc);
    // Record the fullscreen composite that samples the frame's storage image. Must be
    // recorded INSIDE the raster render pass, before the ImGui HUD.
    void recordComposite(VkCommandBuffer cmd, uint32_t frame_idx);

    // Full teardown (context-lifetime objects). Frame resources must be destroyed first.
    void destroy();
};

} // namespace mimir
