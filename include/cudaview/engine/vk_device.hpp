#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#include "cudaview/deletion_queue.hpp"
#include "vk_buffer.hpp"
#include "vk_texture.hpp"

struct VulkanQueue
{
  uint32_t family_index = ~0u;
  VkQueue queue = VK_NULL_HANDLE;
};

struct VulkanDevice
{
  // GPU used for Vulkan operations
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties = {};
  VkPhysicalDeviceFeatures features = {};
  VkPhysicalDeviceMemoryProperties memory_properties = {};

  VkDevice logical_device = VK_NULL_HANDLE;
  VkCommandPool command_pool = VK_NULL_HANDLE;

  VulkanQueue graphics, present;
  DeletionQueue deletors;

  explicit VulkanDevice(VkPhysicalDevice gpu);
  ~VulkanDevice();

  void initLogicalDevice(VkSurfaceKHR surface);
  VkCommandPool createCommandPool(uint32_t queue_idx, VkCommandPoolCreateFlags flags);
  std::vector<VkCommandBuffer> createCommandBuffers(uint32_t buffer_count);
  void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

  VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    const void *extmem_info
  );
  VkDeviceMemory allocateMemory(const VkBuffer buffer,
    VkMemoryPropertyFlags properties, const void *export_info
  );

  VulkanBuffer createBuffer2(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties
  );
  VulkanBuffer createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR handle_type
  );
  VulkanTexture createExternalImage(VkImageType type, VkFormat format, VkExtent3D extent,
    VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags mem_props
  );
  VkSampler createSampler(VkFilter filter, bool enable_anisotropy);
  VkDescriptorPool createDescriptorPool(
    const std::vector<VkDescriptorPoolSize>& sizes
  );
  VkDescriptorSetLayout createDescriptorSetLayout(
    const std::vector<VkDescriptorSetLayoutBinding>& layout_bindings
  );
  void transitionImageLayout(VkImage image, VkFormat format,
    VkImageLayout old_layout, VkImageLayout new_layout
  );
  uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags mem_props);

private:
  VulkanBuffer createBuffer2(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags props, const void *extmem_info, const void *export_info
  );
};
