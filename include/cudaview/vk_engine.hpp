#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cuda_runtime_api.h>

#include <condition_variable> // std::condition_variable
#include <functional> // std::function
#include <map> // std::map
#include <mutex> // std::mutex
#include <thread> // std::thread
#include <vector> // std::vector

enum class DataFormat
{
  Float32,
  Rgba32
};

struct MappedUnstructuredMemory
{
  void *cuda_ptr;
  cudaExternalMemory_t cuda_extmem;
  size_t element_count;
  size_t element_size;
  VkFormat vk_format;
  VkBuffer vk_buffer;
  VkDeviceMemory vk_memory;
};

struct MappedStructuredMemory
{
  void *cuda_ptr;
  cudaExternalMemory_t cuda_extmem;
  size_t element_count;
  size_t element_size;
  VkFormat vk_format;
  VkBuffer vk_buffer;
  VkDeviceMemory vk_memory;
  VkImage vk_image;
  VkImageView vk_view;
  std::array<ulong, 3> extent;
};

class VulkanEngine
{
public:
  VulkanEngine(int2 extent, cudaStream_t stream = 0);
  ~VulkanEngine();
  void init(int width = 800, int height = 600);
  bool toggleRenderingMode(const std::string& key);
  void mainLoop();
  void registerUnstructuredMemory(void **ptr_devmem, size_t elem_count, size_t elem_size);
  void registerStructuredMemory(void **ptr_devmem, size_t width, size_t height,
    size_t elem_size, DataFormat format
  );

  void display(std::function<void(void)> func, size_t iter_count);
  void displayAsync();
  void prepareWindow();
  void updateWindow();
  bool should_resize = false;

private:
  GLFWwindow *window;
  VkInstance instance; // Vulkan library handle
  VkDebugUtilsMessengerEXT debug_messenger; // Vulkan debug output handle
  VkSurfaceKHR surface; // Vulkan window surface
  VkPhysicalDevice physical_device; // GPU used for operations
  VkDevice device;
  VkQueue graphics_queue;
  VkQueue present_queue;
  VkSwapchainKHR swapchain;
  std::vector<VkImage> swapchain_images;
  // How to access the image(s) and which part of it (them) to access
  std::vector<VkImageView> swapchain_views;
  VkFormat swapchain_format;
  VkExtent2D swapchain_extent;
  VkRenderPass render_pass;
  VkDescriptorSetLayout descriptor_layout;
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline, screen_pipeline;
  VkCommandPool command_pool;
  std::vector<VkFramebuffer> framebuffers;
  std::vector<VkCommandBuffer> command_buffers;
  //VkSemaphore vk_presentation_semaphore;
  //VkSemaphore vk_timeline_semaphore;

  std::vector<VkFence> images_inflight;
  std::vector<VkFence> inflight_fences;
  std::vector<VkSemaphore> image_available;
  std::vector<VkSemaphore> render_finished;
  VkSemaphore vk_wait_semaphore;
  VkSemaphore vk_signal_semaphore;

  std::vector<VkBuffer> uniform_buffers;
  std::vector<VkDeviceMemory> ubo_memory;

  VkBuffer vertex_buffer;
  VkDeviceMemory vertex_buffer_memory;
  VkBuffer index_buffer;
  VkDeviceMemory index_buffer_memory;
  uint64_t current_frame;

  std::vector<VkDescriptorSet> descriptor_sets;
  VkSampler texture_sampler;
  // CPU thread synchronization variables
  bool device_working = false;
  std::thread rendering_thread;
  std::mutex mutex;
  std::condition_variable cond;

  VkDescriptorPool imgui_pool, descriptor_pool;
  std::map<std::string, bool> rendering_modes;

  // Cuda interop data
  int2 data_extent;
  std::function<void(void)> step_function;
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_wait_semaphore, cuda_signal_semaphore;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;

  std::vector<MappedStructuredMemory> structured_buffers;
  std::vector<MappedUnstructuredMemory> unstructured_buffers;

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory &memory
  );
  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
  void importExternalBuffer(void *handle, size_t size,
    VkExternalMemoryHandleTypeFlagBits handle_type, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory
  );
  void createExternalBuffer(VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    VkExternalMemoryHandleTypeFlagsKHR handle_type, VkBuffer& buffer,
    VkDeviceMemory& buffer_memory
  );
  void createExternalSemaphore(VkSemaphore& semaphore);
  void createExternalImage(uint32_t width, uint32_t height, VkFormat format,
    VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory
  );
  void transitionImageLayout(VkImage image, VkFormat format,
    VkImageLayout old_layout, VkImageLayout new_layout
  );
  VkImageView createImageView(VkImage image, VkFormat format);
  void *getMemHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer command_buffer);

  void setUnstructuredRendering(VkCommandBuffer& cmd_buffer);

  void getVertexDescriptions(
    std::vector<VkVertexInputBindingDescription>& bind_desc,
    std::vector<VkVertexInputAttributeDescription>& attr_desc
  );
  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);
  void updateUniformBuffer(uint32_t image_index);

  void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
  VkShaderModule createShaderModule(const std::vector<char>& code);

  // Interop import functions
  void importCudaExternalMemory(void **cuda_ptr,
    cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size
  );
  void importCudaExternalSemaphore(
    cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem
  );

  // TODO: Remove
  void createVertexBuffer();
  void createIndexBuffer();
  VkBuffer staging_buffer;
  VkDeviceMemory staging_memory;

  void initVulkan();
  void initImgui();
  void drawFrame();
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
