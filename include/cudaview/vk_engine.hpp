#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <slang-com-ptr.h>
#include <cuda_runtime_api.h>

#include <condition_variable> // std::condition_variable
#include <functional> // std::function
#include <memory> // std::unique_ptr
#include <mutex> // std::mutex
#include <thread> // std::thread
#include <vector> // std::vector

#include "color/color.hpp"

#include "cudaview/deletion_queue.hpp"
#include "cudaview/engine/cudaview.hpp"
#include "cudaview/engine/barrier.hpp"

namespace
{
  static constexpr size_t MAX_FRAMES_IN_FLIGHT = 1;
}

struct Camera;
struct VulkanCudaDevice;
struct VulkanSwapchain;
struct VulkanFramebuffer;

class VulkanEngine
{
public:
  VulkanEngine(cudaStream_t stream = 0);
  ~VulkanEngine();
  void init(int width = 800, int height = 600);
  CudaView addView(void **ptr_devmem, const ViewParams params);

  void display(std::function<void(void)> func, size_t iter_count);
  void displayAsync();
  void prepareWindow();
  void updateWindow();
  bool should_resize = false;

  void setBackgroundColor(color::rgba<float> color);
  void setPointColor(color::rgba<float> color);
  void setEdgeColor(color::rgba<float> color);

private:
  std::unique_ptr<VulkanCudaDevice> dev;
  std::unique_ptr<VulkanSwapchain> swap;
  std::vector<VulkanFramebuffer> fbs;
  GLFWwindow *window = nullptr;
  VkInstance instance = VK_NULL_HANDLE; // Vulkan library handle
  VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE; // Vulkan debug output handle
  VkPhysicalDevice physical_device = VK_NULL_HANDLE; // GPU used for operations
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  std::vector<VkPipeline> pipelines;
  // 0: points2d
  // 1: points3d
  // 2: texture2d
  // 3: texture3d
  // 4: mesh2d
  // 5: mesh3d
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkDescriptorSet> descriptor_sets;
  VkBuffer ubo_buffer = VK_NULL_HANDLE;
  VkDeviceMemory ubo_memory = VK_NULL_HANDLE;
  DeletionQueue deletors;

  // Depth buffer
  VkImage depth_image;
  VkDeviceMemory depth_memory;
  VkImageView depth_view;

  // Synchronization structures
  std::vector<VkFence> images_inflight;
  std::array<FrameBarrier, MAX_FRAMES_IN_FLIGHT> frames;
  InteropBarrier kernel_start, kernel_finish;
  //VkSemaphore vk_presentation_semaphore;
  //VkSemaphore vk_timeline_semaphore;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;

  // CPU thread synchronization variables
  bool device_working = false;
  std::thread rendering_thread;
  std::mutex mutex;
  std::condition_variable cond;

  // Cuda interop data
  uint3 data_extent = {1, 1, 1};
  cudaStream_t stream;
  uint64_t current_frame = 0;
  std::string shader_path;

  std::vector<CudaView> views;

  FrameBarrier& getCurrentFrame();

  void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
  VkShaderModule createShaderModule(const std::vector<char>& code);
  std::vector<VkPipelineShaderStageCreateInfo> compileSlang(
    Slang::ComPtr<slang::ISession> session, const std::string& shader_path,
    const std::vector<std::string>& entry_names
  );

  color::rgba<float> bg_color{.5f, .5f, .5f, 1.f};
  color::rgba<float> point_color{0.f, 0.f, 1.f, 1.f};
  color::rgba<float> edge_color{0.f, 1.f, 0.f, 1.f};

  // Camera functions
  struct {
    bool left = false;
    bool right = false;
    bool middle = false;
  } mouse_buttons;
  bool view_updated = false;
  float2 mouse_pos;
  void handleMouseMove(float x, float y);
  void handleMouseButton(int button, int action);
  void handleScroll(float xoffset, float yoffset);
  static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
  static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
  static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
  static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
  float depth = 0.01f;
  std::unique_ptr<Camera> camera;

  void initVulkan();
  void initImgui();
  void renderFrame();
  void drawObjects(uint32_t image_idx);
  void drawGui();
  void signalKernelFinish();
  void waitKernelStart();

  // Vulkan core-related functions
  void createInstance();
  void pickPhysicalDevice();
  void createSyncObjects();

  // Swapchain-related functions
  void initSwapchain();
  void cleanupSwapchain();
  void recreateSwapchain();
  void createGraphicsPipelines();
  void createBuffers();
  void createDescriptorSets();

  // Update functions
  void updateDescriptorSets();
  void updateUniformBuffer(uint32_t image_index);

  // Depth buffering
  bool hasStencil(VkFormat format);
  VkFormat findDepthFormat();
  VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features
  );

  VkRenderPass createRenderPass();
};
