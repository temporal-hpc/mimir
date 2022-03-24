#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>

#include <condition_variable> // std::condition_variable
#include <functional> // std::function
#include <map> // std::map
#include <memory> // std::unique_ptr
#include <mutex> // std::mutex
#include <thread> // std::thread
#include <vector> // std::vector

#include "color/color.hpp"

#include "cudaview/engine/vk_buffer.hpp"
#include "cudaview/engine/vk_cudamem.hpp"
#include "cudaview/frame.hpp"
#include "cudaview/deletion_queue.hpp"

namespace
{
  static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
}

struct Camera;
struct VulkanCudaDevice;
struct VulkanSwapchain;
struct VulkanFramebuffer;

class VulkanEngine
{
public:
  VulkanEngine(int3 extent = {1, 1, 1}, cudaStream_t stream = 0);
  ~VulkanEngine();
  void init(int width = 800, int height = 600);
  void mainLoop();
  void registerUnstructuredMemory(void **ptr_devmem,
    size_t elem_count, size_t elem_size,
    UnstructuredDataType type, DataDomain domain
  );
  void registerStructuredMemory(void **ptr_devmem,
    uint3 buffer_size, size_t elem_size, DataDomain domain, DataFormat format
  );

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
  VkDevice device = VK_NULL_HANDLE;
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline texture2d_pipeline = VK_NULL_HANDLE;
  VkPipeline texture3d_pipeline = VK_NULL_HANDLE;
  VkPipeline point2d_pipeline = VK_NULL_HANDLE;
  VkPipeline point3d_pipeline = VK_NULL_HANDLE;
  VkPipeline mesh2d_pipeline = VK_NULL_HANDLE;
  VkPipeline mesh3d_pipeline = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkSampler texture_sampler = VK_NULL_HANDLE;
  //std::vector<VkFramebuffer> framebuffers;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkDescriptorSet> descriptor_sets;
  VulkanBuffer ubo;
  DeletionQueue deletors;

  // Synchronization structures
  std::vector<VkFence> images_inflight;
  VkSemaphore vk_wait_semaphore = VK_NULL_HANDLE;
  VkSemaphore vk_signal_semaphore = VK_NULL_HANDLE;
  //VkSemaphore vk_presentation_semaphore;
  //VkSemaphore vk_timeline_semaphore;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;

  // CPU thread synchronization variables
  bool device_working = false;
  std::thread rendering_thread;
  std::mutex mutex;
  std::condition_variable cond;

  // Cuda interop data
  int3 data_extent;
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_wait_semaphore = nullptr;
  cudaExternalSemaphore_t cuda_signal_semaphore = nullptr;

  uint64_t current_frame = 0;
  std::string shader_path;
  std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frames;
  std::vector<MappedStructuredMemory> structured_buffers;
  std::vector<MappedUnstructuredMemory> unstructured_buffers;

  FrameData& getCurrentFrame();

  void getVertexDescriptions2d(
    std::vector<VkVertexInputBindingDescription>& bind_desc,
    std::vector<VkVertexInputAttributeDescription>& attr_desc
  );
  void getVertexDescriptions3d(
    std::vector<VkVertexInputBindingDescription>& bind_desc,
    std::vector<VkVertexInputAttributeDescription>& attr_desc
  );
  void updateUniformBuffer(uint32_t image_index);

  void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
  VkShaderModule createShaderModule(const std::vector<char>& code);

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
  static void framebufferResizeCallback(GLFWwindow *window, int width, int height);
  static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
  static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
  std::unique_ptr<Camera> camera;

  void initVulkan();
  void initImgui();
  void renderFrame();
  void drawObjects(uint32_t image_idx);
  void drawGui();
  void cudaSemaphoreSignal();
  void cudaSemaphoreWait();

  // Vulkan core-related functions
  void createInstance();
  void pickPhysicalDevice();
  void createDescriptorSetLayout();
  void createTextureSampler();
  void createSyncObjects();

  // Swapchain-related functions
  void initSwapchain();
  void cleanupSwapchain();
  void recreateSwapchain();

  // Swapchain subroutines
  void createGraphicsPipelines();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void updateDescriptorSets();
};
