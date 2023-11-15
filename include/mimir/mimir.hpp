#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>

#include <chrono> // std::chrono
#include <functional> // std::function
#include <memory> // std::unique_ptr
#include <thread> // std::thread
#include <vector> // std::vector

#include <mimir/deletion_queue.hpp>
#include <mimir/engine/interop_device.hpp>
#include <mimir/engine/performance_monitor.hpp>

namespace mimir
{

namespace
{
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
}

struct Camera;
struct InteropDevice;
struct VulkanSwapchain;
struct VulkanFramebuffer;

struct AllocatedBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct ViewerOptions
{
    std::string window_title = "Mimir";
    int2 window_size         = { 800, 600 };
    PresentOptions present   = PresentOptions::VSync;
    bool enable_sync         = true;
    bool show_metrics        = false;
    bool fullscreen          = true;
    bool enable_fps_limit    = false;
    int target_fps           = 60;
    uint report_period       = 0;
};

class CudaviewEngine
{

public:
    CudaviewEngine();
    ~CudaviewEngine();
    void init(ViewerOptions opts);
    void init(int width, int height);
    // Main library function, which setups all the visualization interop
    InteropView *createView(void **ptr_devmem, ViewParams params);
    InteropView *getView(uint32_t view_index);
    void loadTexture(InteropView *view, void *data);

    InteropMemory *createBuffer(void **dev_ptr, MemoryParams params);
    InteropView2 *createView(ViewParams2 params);

    void display(std::function<void(void)> func, size_t iter_count);
    void displayAsync();
    void prepareViews();
    void updateViews();
    void showMetrics();
    void exit();

    void setBackgroundColor(float4 color);
    void setGuiCallback(std::function<void(void)> callback) { gui_callback = callback; };
    float getTotalTime() { return total_graphics_time; }

private:
    ViewerOptions options;
    std::unique_ptr<InteropDevice> dev;
    std::unique_ptr<VulkanSwapchain> swap;
    std::vector<VulkanFramebuffer> fbs;
    GLFWwindow *window = nullptr;
    VkInstance instance = VK_NULL_HANDLE; // Vulkan library handle
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE; // Vulkan debug output handle
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkDescriptorSet> descriptor_sets;
    DeletionQueue deletors;
    std::function<void(void)> gui_callback = []() { return; };

    // Depth buffer
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;

    // Synchronization structures
    //std::vector<VkFence> images_inflight;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> frame_fences;
    VkSemaphore present_semaphore = VK_NULL_HANDLE;

    // CPU thread synchronization variables
    bool should_resize = false;
    bool running = false;
    bool kernel_working = false;
    std::thread rendering_thread;
    using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
    chrono_tp last_time = {};

    // Cuda interop data
    InteropBarrier interop;
    uint64_t current_frame = 0;
    std::string shader_path;

    //std::vector<std::unique_ptr<InteropView>> views;
    std::vector<AllocatedBuffer> uniform_buffers;

    std::vector<std::unique_ptr<InteropMemory>> allocations;
    std::vector<std::unique_ptr<InteropView2>> views2;

    float4 bg_color{.5f, .5f, .5f, 1.f};

    // Camera functions
    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouse_buttons;
    bool view_updated = false;
    float2 mouse_pos;
    void handleMouseMove(float x, float y);
    void handleMouseButton(int button, int action, int mods);
    void handleScroll(float xoffset, float yoffset);
    void handleKey(int key, int scancode, int action, int mods);
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
    static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void windowCloseCallback(GLFWwindow *window);

    std::unique_ptr<Camera> camera;
    bool show_demo_window = false;
    long target_frame_time = 0;

    void initVulkan();
    void initImgui();
    void prepare();
    void renderFrame();
    void drawElements(uint32_t image_idx);
    void drawGui();
    void displayEngineGUI();
    void signalKernelFinish();
    void waitKernelStart();

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
    int max_fps = 0;
    PerformanceMonitor perf;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    double total_pipeline_time = 0;
    double getRenderTimeResults(uint32_t cmd_idx);
    std::array<float,240> frame_times{};
    float total_graphics_time = 0;
    size_t total_frame_count = 0;
};

} // namespace mimir