#pragma once

#include <vulkan/vulkan.h>

#include <vector>

struct VulkanDevice
{
  // GPU used for Vulkan operations
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceFeatures features;
  VkPhysicalDeviceMemoryProperties memory_properties;

  VkDevice logical_device = VK_NULL_HANDLE;
  VkCommandPool command_pool = VK_NULL_HANDLE;

  struct
  {
    uint32_t graphics;
    uint32_t present;
  } queue_indices;

  explicit VulkanDevice(VkPhysicalDevice gpu);
  ~VulkanDevice();

  void initLogicalDevice(VkSurfaceKHR surface);
  VkCommandPool createCommandPool(uint32_t queue_idx, VkCommandPoolCreateFlags flags);
  std::vector<VkCommandBuffer> createCommandBuffers(uint32_t buffer_count);
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer command_buffer, VkQueue queue);

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory
  );
  void createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR handle_type,
    VkBuffer& buffer, VkDeviceMemory& memory
  );
  void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size, VkQueue queue);

  uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties);
};
