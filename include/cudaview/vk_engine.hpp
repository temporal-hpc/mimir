#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <map> // std::map
#include <vector> // std::vector

struct SwapChainSupportDetails
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
  void *getMemHandle(VkDeviceMemory memory,
    VkExternalMemoryHandleTypeFlagBits handle_type
  );
  void *getSemaphoreHandle(VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handle_type
  );
  bool toggleRenderingMode(const std::string& key);
  void mainLoop();
  bool should_resize = false;
  size_t element_count;

protected:
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
  VkPipelineLayout pipeline_layout, screen_layout;
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

  VkImage texture_image;
  VkImageView texture_view;

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
  void createTextureImage();
  void createTextureImageView();
  void createTextureSampler();
  void updateDescriptorsUnstructured();
  void updateDescriptorsStructured();
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

  virtual void setUnstructuredRendering(VkCommandBuffer& cmd_buffer,
    uint32_t vertex_count
  );
  virtual std::vector<const char*> getRequiredExtensions() const;
  virtual std::vector<const char*> getRequiredDeviceExtensions() const;
  virtual void getVertexDescriptions(
    std::vector<VkVertexInputBindingDescription>& bind_desc,
    std::vector<VkVertexInputAttributeDescription>& attr_desc
  );
  virtual void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);
  virtual void updateUniformBuffer(uint32_t image_index);
  virtual void drawFrame();
  virtual void initVulkan();

  virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;

private:
  GLFWwindow *window;
  VkDescriptorPool imgui_pool, descriptor_pool;
  std::vector<VkDescriptorSet> descriptor_sets;
  std::map<std::string, bool> rendering_modes;

  VkBuffer staging_buffer;
  VkDeviceMemory staging_memory;
  VkSampler texture_sampler;
  VkDeviceMemory texture_memory;

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

  void initSwapchain();

  void createGraphicsPipeline(
    const std::string& vertex_file, const std::string& fragment_file
  );
  void createTextureGraphicsPipeline(const std::string& vertex_file,
    const std::string& fragment_file
  );
  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

  void cleanupSwapchain();
  void recreateSwapchain();

  bool checkAllExtensionsSupported(VkPhysicalDevice device,
    const std::vector<const char*>& device_extensions) const;
  bool isDeviceSuitable(VkPhysicalDevice device) const;
  bool findQueueFamilies(VkPhysicalDevice device,
    uint32_t& graphics_family, uint32_t& present_family) const;
  SwapChainSupportDetails getSwapchainProperties(VkPhysicalDevice device) const;

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
  VkShaderModule createShaderModule(const std::vector<char>& code);
  void createVertexBuffer();
  void createIndexBuffer();
};
