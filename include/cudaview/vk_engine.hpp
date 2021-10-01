#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "cudaview/vk_types.hpp"

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
  void createExternalSemaphore(VkSemaphore& semaphore);
  void mainLoop();
  bool should_resize = false;

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
  VkPipelineLayout pipeline_layout;
  VkPipeline graphics_pipeline;
  VkCommandPool command_pool;
  std::vector<VkFramebuffer> framebuffers;
  std::vector<VkCommandBuffer> command_buffers;
  VkSemaphore vk_presentation_semaphore;
  VkSemaphore vk_timeline_semaphore;
  VkBuffer vertex_buffer;
  VkDeviceMemory vertex_buffer_memory;
  VkBuffer index_buffer;
  VkDeviceMemory index_buffer_memory;
  size_t current_frame;

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory &memory
  );
  void createExternalBuffer(VkDeviceSize size,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    VkExternalMemoryHandleTypeFlagsKHR handle_type, VkBuffer& buffer,
    VkDeviceMemory& buffer_memory
  );
  void importExternalBuffer(void *handle, size_t size,
    VkExternalMemoryHandleTypeFlagBits handle_type, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory
  );
  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);

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
  virtual void drawFrame();
  virtual void initApplication();

private:
  GLFWwindow *window;
  void initVulkan();
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createInstance();
  void createSurface();
  void createSwapChain();
  void createImageViews();
  void createRenderPass();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSyncObjects();

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
