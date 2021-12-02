#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <condition_variable> // std::condition_variable
#include <map> // std::map
#include <mutex> // std::mutex
#include <thread> // std::thread
#include <vector> // std::vector

struct SwapchainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

class VulkanEngine
{
public:
  VulkanEngine();
  virtual ~VulkanEngine();
  void init(int width = 800, int height = 600);
  bool toggleRenderingMode(const std::string& key);
  void mainLoop();
  bool should_resize = false;

protected:
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
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer command_buffer);
  void createTextureSampler();
  void createImage(uint32_t width, uint32_t height, VkFormat format,
    VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory
  );
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
  void drawFrame();
  void drawGui();
  void initSwapchain();
  void cleanupSwapchain();

  virtual void setUnstructuredRendering(VkCommandBuffer& cmd_buffer);
  virtual std::vector<const char*> getRequiredExtensions() const;
  virtual std::vector<const char*> getRequiredDeviceExtensions() const;
  virtual void getVertexDescriptions(
    std::vector<VkVertexInputBindingDescription>& bind_desc,
    std::vector<VkVertexInputAttributeDescription>& attr_desc
  );
  virtual void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);
  virtual void updateUniformBuffer(uint32_t image_index);
  virtual void initVulkan();

  virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
  virtual void recreateSwapchain();

  std::vector<VkDescriptorSet> descriptor_sets;
  VkSampler texture_sampler;
  // CPU thread synchronization variables
  bool device_working = false;
  std::thread rendering_thread;
  std::mutex mutex;
  std::condition_variable cond;

private:
  VkDescriptorPool imgui_pool, descriptor_pool;
  std::map<std::string, bool> rendering_modes;

  void initImgui();
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createInstance();
  void createSurface();
  void createSwapChain();
  void createImageViews();
  void createRenderPass();
  void createGraphicsPipelines();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSyncObjects();
  void createDescriptorSetLayout();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

  bool checkAllExtensionsSupported(VkPhysicalDevice device,
    const std::vector<const char*>& device_extensions) const;
  bool isDeviceSuitable(VkPhysicalDevice device) const;
  bool findQueueFamilies(VkPhysicalDevice device,
    uint32_t& graphics_family, uint32_t& present_family) const;
  SwapchainSupportDetails getSwapchainProperties(VkPhysicalDevice device) const;

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
  VkShaderModule createShaderModule(const std::vector<char>& code);

  // TODO: Remove
  void createVertexBuffer();
  void createIndexBuffer();
  VkBuffer staging_buffer;
  VkDeviceMemory staging_memory;
};
