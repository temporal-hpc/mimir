#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "cudaview/vk_types.hpp"

#include <iostream> // std::cerr

struct SwapChainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

class VulkanEngine
{
public:
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
  GLFWwindow *window = nullptr;
  VkInstance instance = VK_NULL_HANDLE; // Vulkan library handle
  VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE; // Vulkan debug output handle
  VkSurfaceKHR surface = VK_NULL_HANDLE; // Vulkan window surface
  VkPhysicalDevice physical_device = VK_NULL_HANDLE; // GPU used for operations
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue present_queue = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swapchain_images;
  // How to access the image(s) and which part of it (them) to access
  std::vector<VkImageView> swapchain_views;
  VkFormat swapchain_format;
  VkExtent2D swapchain_extent;
  VkRenderPass render_pass;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline graphics_pipeline = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkSemaphore> image_available;
  std::vector<VkSemaphore> render_finished;
  std::vector<VkFence> inflight_fences;
  std::vector<VkFence> images_inflight;
  size_t current_frame = 0;
  VkSemaphore vk_timeline_semaphore = VK_NULL_HANDLE;
  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VkDeviceMemory vertex_buffer_memory = VK_NULL_HANDLE;
  VkBuffer index_buffer = VK_NULL_HANDLE;
  VkDeviceMemory index_buffer_memory = VK_NULL_HANDLE;

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

  virtual void initApplication() {}
  virtual void setUnstructuredRendering(VkCommandBuffer& cmd_buffer,
    uint32_t vertex_count
  );
  virtual std::vector<const char*> getRequiredExtensions() const;
  virtual std::vector<const char*> getRequiredDeviceExtensions() const;
  virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkVertexInputAttributeDescription>& attr_desc) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
  virtual void drawFrame();

private:
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
