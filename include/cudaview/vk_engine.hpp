#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>

#include <chrono> // std::chrono
#include <functional> // std::function
#include <memory> // std::unique_ptr
#include <thread> // std::thread
#include <vector> // std::vector

#include <cudaview/deletion_queue.hpp>
#include <cudaview/engine/vk_cudadevice.hpp>

namespace
{
     static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
}

struct Camera;
struct VulkanCudaDevice;
struct VulkanSwapchain;
struct VulkanFramebuffer;

struct AllocatedBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct ViewerOptions
{
    int2 window            = { 800, 600 };
    PresentOptions present = PresentOptions::TripleBuffering;
    bool show_metrics      = false;
    uint report_period     = 5;
};

class VulkanEngine
{

public:
    VulkanEngine();
    ~VulkanEngine();
    void init(ViewerOptions opts);
    void init(int width, int height);
    // Main library function, which setups all the visualization interop
    CudaView *createView(void **ptr_devmem, ViewParams params);
    CudaView *getView(uint32_t view_index);
    void loadTexture(CudaView *view, void *data);

    void display(std::function<void(void)> func, size_t iter_count);
    void displayAsync();
    void prepareWindow();
    void updateWindow();
    void showMetrics();
    bool should_resize = false;

    void setBackgroundColor(float4 color);

private:
    ViewerOptions options;
    std::unique_ptr<VulkanCudaDevice> dev;
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

    // Depth buffer
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_memory = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;

    // Synchronization structures
    //std::vector<VkFence> images_inflight;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> frame_fences;
    VkSemaphore present_semaphore = VK_NULL_HANDLE;
    InteropBarrier timeline;

    // CPU thread synchronization variables
    bool running = false;
    bool kernel_working = false;
    std::thread rendering_thread;
    using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
    chrono_tp last_time = {};

    // Cuda interop data
    cudaStream_t stream = 0; // TODO: Remove
    uint64_t current_frame = 0;
    std::string shader_path;

    std::vector<CudaView> views;
    std::vector<AllocatedBuffer> uniform_buffers;

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

    static void addViewObjectGui(CudaView *view_ptr, int uid);
    std::unique_ptr<Camera> camera;
    bool show_demo_window = false;

    void initVulkan();
    void initImgui();
    void prepare();
    void renderFrame();
    void drawObjects(uint32_t image_idx);
    void drawGui();
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
    void rebuildPipeline(CudaView& view);
    void initUniformBuffers();
    void updateUniformBuffers(uint32_t image_idx);

    // Depth buffering
    bool hasStencil(VkFormat format);
    VkFormat findDepthFormat();
};
