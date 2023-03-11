#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

#include "cudaview/deletion_queue.hpp"

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

  uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags mem_props);
  VkDeviceMemory allocateMemory(VkMemoryRequirements requirements,
    VkMemoryPropertyFlags properties, const void *export_info = nullptr
  );
  VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    const void *extmem_info = nullptr
  );
  VkImage createImage(VkImageType type, VkFormat format, VkExtent3D extent,
    VkImageTiling tiling, VkImageUsageFlags usage, const void *extmem_info = nullptr
  );
  VkSampler createSampler(VkFilter filter, bool enable_anisotropy);
  void generateMipmaps(VkImage image, VkFormat img_format,
    int img_width, int img_height, int mip_levels
  );
  VkDescriptorPool createDescriptorPool(
    const std::vector<VkDescriptorPoolSize>& sizes
  );
  // Get memory requirements for later allocation
  // TODO: Use span instead of vector
  VkMemoryRequirements getMemoryRequiements(VkBufferUsageFlags usage,
    const std::vector<uint32_t>& sizes
  );
  std::vector<VkDescriptorSet> createDescriptorSets(
    VkDescriptorPool pool, VkDescriptorSetLayout layout, uint32_t set_count
  );
  VkDescriptorSetLayout createDescriptorSetLayout(
    const std::vector<VkDescriptorSetLayoutBinding>& layout_bindings
  );
  void transitionImageLayout(VkImage image,
    VkImageLayout old_layout, VkImageLayout new_layout
  );
};
