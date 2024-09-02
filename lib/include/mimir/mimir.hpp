#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <chrono> // std::chrono
#include <functional> // std::function
#include <memory> // std::unique_ptr
#include <thread> // std::thread
#include <vector> // std::vector

#include <mimir/engine/interop_view.hpp>
#include <mimir/engine/interop_device.hpp>
#include <mimir/engine/performance_monitor.hpp>
#include <mimir/deletion_queue.hpp>

namespace mimir
{

namespace
{
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    // Timeout value for frame acquisition and synchronization structures
    // To remove the timeout, use std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t frame_timeout = 1000000000;
}

struct Camera;
struct GlfwContext;
struct InteropDevice;
struct InteropBarrier;
struct VulkanSwapchain;
struct VulkanFramebuffer;

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

class MimirEngine
{

public:
    MimirEngine(); // TODO: Replace with factory
    ~MimirEngine();
    void init(ViewerOptions opts);
    void init(int width, int height);
    // Main library function, which setups all the visualization interop
    InteropView *createView(ViewParams params);
    InteropMemory *createBuffer(void **dev_ptr, MemoryParams params);

    // Allocates linear device memory, equivalent to cudaMalloc(dev_ptr, size)
    std::shared_ptr<Allocation> allocLinear(void **dev_ptr, size_t size);
    // Allocates opaque read-only device memory, equivalent to cudaCreateTextureObject()
    std::shared_ptr<Allocation> allocTexture(cudaTextureObject_t *tex_obj, const cudaResourceDesc *res_desc,
        const cudaTextureDesc *tex_desc, const cudaResourceViewDesc *view_desc
    );
    // Allocates opaque RW device memory, equivalent to cudaCreateSurfaceObject()
    std::shared_ptr<Allocation> allocSurface(cudaSurfaceObject_t *surf_obj, const cudaResourceDesc *res_desc);

    // Creates an allocation with memory initialized for representing a structured domain
    std::shared_ptr<Allocation> makeStructuredDomain(StructuredDomainParams p);

    // View creation
    std::shared_ptr<InteropView2> createView(ViewParams2 params);

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
    bool running = false;
    bool should_resize = false;

    // Camera functions
    std::unique_ptr<Camera> camera;
    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouse_buttons;
    bool view_updated = false;
    float2 mouse_pos;

    void signalKernelFinish();

private:
    VkBuffer initBuffer();

    VkInstance instance                     = VK_NULL_HANDLE;
    VkRenderPass render_pass                = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout        = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool        = VK_NULL_HANDLE;

    InteropDevice dev;
    //VmaAllocator allocator = nullptr;
    //VmaPool interop_pool   = nullptr;

    std::unique_ptr<VulkanSwapchain> swap;
    std::vector<VulkanFramebuffer> fbs;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkDescriptorSet> descriptor_sets;
    std::function<void(void)> gui_callback = []() { return; };

    // Depth buffer
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;

    // Synchronization structures
    std::array<SyncData, MAX_FRAMES_IN_FLIGHT> sync_data;

    // CPU thread synchronization variables
    bool kernel_working = false;
    std::thread rendering_thread;
    using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
    chrono_tp last_time = {};

    // Cuda interop data
    std::unique_ptr<InteropBarrier> interop;
    std::string shader_path;
    uint64_t render_timeline = 0;
    long target_frame_time = 0;

    std::vector<AllocatedBuffer> uniform_buffers;
    std::vector<InteropMemory*> allocations;
    std::vector<std::shared_ptr<InteropView2>> views;
    std::unique_ptr<GlfwContext> window_context;

    // Deletion queues organized by lifetime
    struct {
        DeletionQueue context;
        DeletionQueue swapchain;
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
    void pickPhysicalDevice();
    void createSyncObjects();
    void updateDescriptorSets();
    VkRenderPass createRenderPass();

    // Swapchain-related functions
    void initSwapchain();
    void cleanupSwapchain();
    void recreateSwapchain();
    void createGraphicsPipelines();
    void rebuildPipeline(InteropView& view);
    void initUniformBuffers();
    void updateUniformBuffers(uint32_t image_idx);

    // Depth buffering
    bool hasStencil(VkFormat format);
    VkFormat findDepthFormat();

    // Benchmarking
    PerformanceMonitor perf;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    double total_pipeline_time = 0;
    double getRenderTimeResults(uint32_t cmd_idx);
    std::array<float,240> frame_times{};
    float total_graphics_time = 0;
    size_t total_frame_count = 0;
};

} // namespace mimir