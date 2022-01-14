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

#include "cudaview/vk_cuda_map.hpp"
#include "cudaview/frame.hpp"

namespace
{
  static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
}

struct Camera;

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
    size_t width, size_t height, size_t elem_size, DataFormat format
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
  GLFWwindow *window;
  VkInstance instance; // Vulkan library handle
  VkDebugUtilsMessengerEXT debug_messenger; // Vulkan debug output handle
  VkSurfaceKHR surface; // Vulkan window surface
  VkPhysicalDevice physical_device; // GPU used for operations
  VkPhysicalDeviceProperties device_properties;
  VkDevice device;
  VkQueue graphics_queue, present_queue;
  VkSwapchainKHR swapchain;
  std::vector<VkImage> swapchain_images;
  // How to access the image(s) and which part of it (them) to access
  std::vector<VkImageView> swapchain_views;
  VkFormat swapchain_format;
  VkExtent2D swapchain_extent;
  VkRenderPass render_pass;
  VkDescriptorSetLayout descriptor_layout;
  VkPipelineLayout pipeline_layout;
  VkPipeline screen_pipeline;
  VkPipeline point2d_pipeline, point3d_pipeline;
  VkPipeline mesh2d_pipeline, mesh3d_pipeline;
  VkCommandPool command_pool;
  VkDescriptorPool descriptor_pool;
  VkSampler texture_sampler;
  std::vector<VkFramebuffer> framebuffers;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkDescriptorSet> descriptor_sets;
  VkBuffer uniform_buffer;
  VkDeviceMemory ubo_memory;

  // Synchronization structures
  std::vector<VkFence> images_inflight;
  VkSemaphore vk_wait_semaphore;
  VkSemaphore vk_signal_semaphore;
  //VkSemaphore vk_presentation_semaphore;
  //VkSemaphore vk_timeline_semaphore;

  // CPU thread synchronization variables
  bool device_working = false;
  std::thread rendering_thread;
  std::mutex mutex;
  std::condition_variable cond;

  // Cuda interop data
  int3 data_extent;
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_wait_semaphore, cuda_signal_semaphore;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;

  uint64_t current_frame;
  std::string shader_path;
  std::array<FrameData, MAX_FRAMES_IN_FLIGHT> frames;
  std::vector<MappedStructuredMemory> structured_buffers;
  std::vector<MappedUnstructuredMemory> unstructured_buffers;

  FrameData& getCurrentFrame();

  // Buffer functions
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory &memory
  );
  void createExternalBuffer(VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    VkExternalMemoryHandleTypeFlagsKHR handle_type, VkBuffer& buffer,
    VkDeviceMemory& buffer_memory
  );
  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);

  void createExternalSemaphore(VkSemaphore& semaphore);
  void createExternalImage(uint32_t width, uint32_t height, VkFormat format,
    VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory
  );
  void transitionImageLayout(VkImage image, VkFormat format,
    VkImageLayout old_layout, VkImageLayout new_layout
  );
  VkImageView createImageView(VkImage image, VkFormat format);
  void *getMemoryHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer command_buffer);

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

  // Interop import functions
  void importCudaExternalMemory(void **cuda_ptr,
    cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size
  );
  void importCudaExternalSemaphore(
    cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem
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
  void createCoreObjects();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createCommandPool();
  void createDescriptorSetLayout();
  void createTextureSampler();
  void createSyncObjects();

  // Swapchain-related functions
  void initSwapchain();
  void cleanupSwapchain();
  void recreateSwapchain();

  // Swapchain subroutines
  void createSwapchain();
  void createImageViews();
  void createRenderPass();
  void createGraphicsPipelines();
  void createFramebuffers();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void createCommandBuffers();
  void updateDescriptorSets();
};
