#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <functional> // std::function
#include <thread> // std::thread
#include <vector> // std::vector

#include <mimir/options.hpp>
#include <mimir/view.hpp>

#include "api.hpp"
#include "camera.hpp"
#include "deletion_queue.hpp"
#include "device.hpp"
#include "framebuffer.hpp"
#include "interop.hpp"
#include "metrics.hpp"
#include "pipeline.hpp"
#include "swapchain.hpp"
#include "window.hpp"

namespace mimir
{

namespace
{
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    // Timeout value for frame acquisition and synchronization structures
    // To remove the timeout, use std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t frame_timeout = 2000000000;
}

struct AllocatedBuffer
{
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct SyncData
{
    VkFence frame_fence;
    VkSemaphore image_acquired;
    VkSemaphore render_complete;
};

struct VulkanQueue
{
    uint32_t family_index;
    VkQueue queue;
};

struct MimirEngine
{
    ViewerOptions options;

    VkInstance instance;
    PhysicalDevice physical_device;
    VulkanQueue graphics, present;
    VkDevice device;
    VkCommandPool command_pool;

    VkRenderPass render_pass;
    VkDescriptorSetLayout descriptor_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool descriptor_pool;
    VkSurfaceKHR surface;

    Swapchain swapchain;
    PipelineBuilder pipeline_builder;
    //VmaAllocator allocator = nullptr;
    //VmaPool interop_pool   = nullptr;

    Framebuffer framebuffers;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkDescriptorSet> descriptor_sets;
    std::function<void(void)> gui_callback;

    // Depth buffer
    VkImage depth_image;
    VkDeviceMemory depth_memory;
    VkImageView depth_view;

    // Synchronization structures
    std::array<SyncData, MAX_FRAMES_IN_FLIGHT> sync_data;
    interop::Barrier interop;

    uint64_t render_timeline;
    bool running;
    bool compute_active;
    std::thread rendering_thread;

    std::vector<AllocatedBuffer> uniform_buffers;
    std::vector<View*> views;
    GlfwContext window_context;
    Camera camera;

    // Deletion queues organized by lifetime
    struct {
        DeletionQueue context;
        DeletionQueue graphics;
        DeletionQueue views;
    } deletors;

    // Benchmarking
    metrics::GraphicsMonitor graphics_monitor;
    metrics::ComputeMonitor compute_monitor;

    static MimirEngine make(ViewerOptions opts);
    static MimirEngine make(int width, int height);

    // Allocates linear device memory, equivalent to cudaMalloc(dev_ptr, size)
    LinearAlloc *allocLinear(void **dev_ptr, size_t size);
    // Allocates opaque device memory, equivalent to cudaMallocMipmappedArray()
    OpaqueAlloc *allocMipmap(cudaMipmappedArray_t *dev_arr,
         const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels = 1
    );

    // Allocates device memory initialized for representing a structured domain
    AttributeDescription makeStructuredGrid(Layout size, float3 start={0.f,0.f,0.f});
    AttributeDescription makeImageDomain();

    // View creation
    View *createView(ViewDescription *desc);
    VkBuffer createAttributeBuffer(VkDeviceSize size,
        VkBufferUsageFlags usage, VkDeviceMemory memory
    );

    void display(std::function<void(void)> func, size_t iter_count);
    void displayAsync();
    void prepareViews();
    void updateViews();
    void deinit();
    void exit();
    PerformanceMetrics getMetrics();

    void setGuiCallback(std::function<void(void)> callback) { gui_callback = callback; };

    void initVulkan();
    void prepare();
    void renderFrame();
    void drawElements(uint32_t image_idx);
    void waitKernelStart();
    void signalKernelFinish();
    void waitTimelineHost();

    // Vulkan core-related functions
    void createInstance();
    void createSyncObjects();
    void updateDescriptorSets();

    // Swapchain-related functions
    void initGraphics();
    void cleanupGraphics();
    void recreateGraphics();
    void createViewPipelines();
    void initUniformBuffers();
    void updateUniformBuffers(uint32_t image_idx);

    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);
    void loadTexture(TextureDescription desc, void *img_data, size_t memsize);
    void copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent);
    void generateMipmaps(VkImage image, VkFormat img_format,
        int img_width, int img_height, int mip_levels
    );
    void transitionImageLayout(VkImage image,
        VkImageLayout old_layout, VkImageLayout new_layout
    );
};

static_assert(std::is_default_constructible_v<MimirEngine>);
//static_assert(std::is_nothrow_default_constructible_v<MimirEngine>);
//static_assert(std::is_trivially_default_constructible_v<MimirEngine>);

} // namespace mimir