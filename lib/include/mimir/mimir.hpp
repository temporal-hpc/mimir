#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <chrono> // std::chrono
#include <functional> // std::function
#include <thread> // std::thread
#include <vector> // std::vector

#include <mimir/engine/camera.hpp>
#include <mimir/engine/deletion_queue.hpp>
#include <mimir/engine/device.hpp>
#include <mimir/engine/framebuffer.hpp>
#include <mimir/engine/interop.hpp>
#include <mimir/engine/interop_view.hpp>
#include <mimir/engine/performance_monitor.hpp>
#include <mimir/engine/swapchain.hpp>
#include <mimir/engine/window.hpp>

namespace mimir
{

namespace
{
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    // Timeout value for frame acquisition and synchronization structures
    // To remove the timeout, use std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t frame_timeout = 1000000000;
}

struct AllocatedBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct SyncData
{
    VkFence frame_fence = VK_NULL_HANDLE;
    VkSemaphore image_acquired = VK_NULL_HANDLE;
    VkSemaphore render_complete = VK_NULL_HANDLE;
};

struct WindowOptions
{
    std::string title = "Mimir";
    int2 size         = { 800, 600 };
    bool fullscreen   = false; // TODO: Implement
};

struct PresentOptions
{
    PresentMode mode      = PresentMode::Immediate;
    bool enable_sync      = true;
    bool enable_fps_limit = false;
    int target_fps        = 60;
    int max_fps           = 0;
};

struct ViewerOptions
{
    WindowOptions window   = {};
    PresentOptions present = {};
    bool show_metrics      = false;
    bool show_demo_window  = false;
    uint report_period     = 0;
    float4 bg_color        = {.5f, .5f, .5f, 1.f};
};

struct VulkanQueue
{
    uint32_t family_index = ~0u;
    VkQueue queue         = VK_NULL_HANDLE;
};

struct MimirEngine
{

    static MimirEngine make(ViewerOptions opts);
    static MimirEngine make(int width, int height);
    // Main library function, which setups all the visualization interop
    InteropViewOld *createView(ViewParamsOld params);
    InteropMemory *createBufferOld(void **dev_ptr, MemoryParams params);

    // Allocates linear device memory, equivalent to cudaMalloc(dev_ptr, size)
    std::shared_ptr<Allocation> allocLinear(void **dev_ptr, size_t size);
    // Allocates opaque device memory, equivalent to cudaMallocMipmappedArray()
    std::shared_ptr<Allocation> allocMipmap(cudaMipmappedArray_t *dev_arr,
        const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels = 1
    );

    // Allocates device memory initialized for representing a structured domain
    AttributeParams makeStructuredDomain(StructuredDomainParams p);

    // View creation
    std::shared_ptr<InteropView> createView(ViewParams params);

    void loadTexture(InteropMemory *interop, void *data);
    void display(std::function<void(void)> func, size_t iter_count);
    void displayAsync();
    void prepareViews();
    void updateViews();
    void showMetrics();
    void exit();

    void setGuiCallback(std::function<void(void)> callback) { gui_callback = callback; };
    float getTotalTime() { return total_graphics_time; }

    ViewerOptions options;
    bool running;

    // Camera functions
    Camera camera;
    bool view_updated;

    VkInstance instance;
    PhysicalDevice physical_device{};
    VulkanQueue graphics, present;
    VkDevice device;
    VkCommandPool command_pool;

    VkRenderPass render_pass;
    VkDescriptorSetLayout descriptor_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool descriptor_pool;
    VkSurfaceKHR surface;

    Swapchain swapchain;
    //VmaAllocator allocator = nullptr;
    //VmaPool interop_pool   = nullptr;

    std::vector<VulkanFramebuffer> fbs;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkDescriptorSet> descriptor_sets;
    std::function<void(void)> gui_callback;

    // Depth buffer
    VkImage depth_image;
    VkDeviceMemory depth_memory;
    VkImageView depth_view;

    // Synchronization structures
    std::array<SyncData, MAX_FRAMES_IN_FLIGHT> sync_data;

    // CPU thread synchronization variables
    bool kernel_working;
    std::thread rendering_thread;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_time;

    // Cuda interop data
    interop::Barrier interop;
    std::string shader_path;
    uint64_t render_timeline;
    long target_frame_time;

    std::vector<AllocatedBuffer> uniform_buffers;
    std::vector<InteropMemory*> allocations;
    std::vector<std::shared_ptr<InteropView>> views;
    GlfwContext window_context;

    // Deletion queues organized by lifetime
    struct {
        DeletionQueue context;
        DeletionQueue graphics;
        DeletionQueue views;
    } deletors;

    void updateLinearTextures();
    void listExtensions();
    void initVulkan();
    void prepare();
    void renderFrame();
    void drawElements(uint32_t image_idx);
    void waitKernelStart();
    void waitTimelineHost();

    // Vulkan core-related functions
    void createInstance();
    void createSyncObjects();
    void updateDescriptorSets();
    VkRenderPass createRenderPass();

    // Swapchain-related functions
    void initGraphics();
    void cleanupGraphics();
    void recreateGraphics();
    void createViewPipelines();
    void rebuildPipeline(InteropViewOld& view);
    void initUniformBuffers();
    void updateUniformBuffers(uint32_t image_idx);

    // Depth buffering
    VkFormat findDepthFormat();

    void signalKernelFinish();

    Allocation allocExtmemBuffer(size_t size, VkBufferUsageFlags usage);
    VkBuffer createAttributeBuffer(const AttributeParams attr, size_t element_count, VkBufferUsageFlags usage);

    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

    void generateMipmaps(VkImage image, VkFormat img_format,
        int img_width, int img_height, int mip_levels
    );
    void transitionImageLayout(VkImage image,
        VkImageLayout old_layout, VkImageLayout new_layout
    );

    // Benchmarking
    PerformanceMonitor perf;
    VkQueryPool query_pool;
    double total_pipeline_time;
    double getRenderTimeResults(uint32_t cmd_idx);
    std::array<float,240> frame_times;
    float total_graphics_time;
    size_t total_frame_count;
};

static_assert(std::is_default_constructible_v<MimirEngine>);
//static_assert(std::is_nothrow_default_constructible_v<MimirEngine>);
//static_assert(std::is_trivially_default_constructible_v<MimirEngine>);

} // namespace mimir