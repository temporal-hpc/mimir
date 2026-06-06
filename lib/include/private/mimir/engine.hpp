#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <functional> // std::function
#include <thread> // std::thread
#include <vector> // std::vector

#include <mimir/options.hpp>
#include <mimir/remote_protocol.hpp>
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

struct MimirInstance
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

    // Offscreen color targets used in headless mode (in place of swapchain images).
    std::vector<VkImage> offscreen_images;
    std::vector<VkDeviceMemory> offscreen_memory;
    // Index of the most recently rendered image (used to read back headless frames).
    uint32_t last_image_idx;

    // Persistent device-local buffer (BGRA, packed) exported to CUDA, used by mapFrameToCuda()
    // for zero-copy NVENC: the rendered image is copied here GPU->GPU and handed to the encoder
    // as a CUDA device pointer, so pixels never touch host memory. Created lazily.
    VkBuffer            frame_cuda_buf_    = VK_NULL_HANDLE;
    VkDeviceMemory      frame_cuda_mem_    = VK_NULL_HANDLE;
    cudaExternalMemory_t frame_cuda_extmem_ = nullptr;
    void               *frame_cuda_ptr_    = nullptr;

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

    static MimirInstance make(ViewerOptions opts);
    static MimirInstance make(int width, int height);

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

    // True when this instance renders offscreen with no window/surface.
    bool isHeadless() const { return options.render_mode != RenderMode::Local; }

    // Renders iter_count frames with no window, calling func before each frame
    // (e.g. to advance a simulation). The last frame can be read back with saveFrameToPpm().
    void renderHeadless(std::function<void(void)> func, size_t iter_count);
    // Copies the most recently rendered offscreen frame into out (B8G8R8A8 bytes).
    void readFrameBytes(std::vector<unsigned char>& out);
    // Copies the most recently rendered offscreen frame into a persistent CUDA-mapped device
    // buffer (BGRA, packed, stride = width*4) entirely on the GPU and returns its CUDA device
    // pointer (nullptr on failure). For zero-copy NVENC; the buffer is created on first use.
    void *mapFrameToCuda();
    // Writes the most recently rendered offscreen frame to a binary PPM (P6) file.
    void saveFrameToPpm(const char *path);
    // Streams rendered frames over TCP to a single connected client and applies the
    // control events it sends back (camera, pause). Renders headless; blocks until the
    // client disconnects, sends Quit, or max_iters compute steps elapse (0 = unlimited).
    void serveRemote(uint16_t port, std::function<void(void)> compute, size_t max_iters,
        bool use_h264 = false,
        remote::TransportKind kind = remote::TransportKind::Tcp);

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
    // Creates the offscreen color image ring used in headless mode.
    void createOffscreenTarget(int width, int height);
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

static_assert(std::is_default_constructible_v<MimirInstance>);
//static_assert(std::is_nothrow_default_constructible_v<MimirInstance>);
//static_assert(std::is_trivially_default_constructible_v<MimirInstance>);

} // namespace mimir