#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "cudaview/vk_types.hpp"

#include <functional> // std::function
#include <iostream> // std::cerr
#include <optional> // std::optional

// Required device extensions
const std::vector<const char*> device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices
{
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;
  bool isComplete()
  {
    return graphics_family.has_value() && present_family.has_value();
  }
};

struct SwapChainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

class VulkanEngine
{
public:
  bool should_resize = false;
  void init(int width = 800, int height = 600);
  void mainLoop();
  virtual ~VulkanEngine();

protected:
  GLFWwindow *window = nullptr;
  VkInstance instance; // Vulkan library handle
  VkDebugUtilsMessengerEXT debug_messenger; // Vulkan debug output handle
  VkSurfaceKHR surface; // Vulkan window surface
  VkPhysicalDevice physical_device = VK_NULL_HANDLE; // GPU used for operations
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
  std::vector<VkFramebuffer> framebuffers;
  VkCommandPool command_pool;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkSemaphore> image_available;
  std::vector<VkSemaphore> render_finished;
  std::vector<VkFence> inflight_fences;
  std::vector<VkFence> images_inflight;
  size_t current_frame = 0;
  VkBuffer vertex_buffer;
  VkDeviceMemory vertex_buffer_memory;
  VkBuffer index_buffer;
  VkDeviceMemory index_buffer_memory;

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
  bool checkDeviceExtensionSupport(VkPhysicalDevice device);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
  VkShaderModule createShaderModule(const std::vector<char>& code);
  void createVertexBuffer();
  void createIndexBuffer();

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

  virtual std::vector<const char*> getRequiredExtensions() const;
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

  bool isDeviceSuitable(VkPhysicalDevice device);
};
