#pragma once

#include <vulkan/vulkan.h>

#include <span> // std::span
#include <vector> // std::vector

namespace mimir
{

struct ImageParams
{
    VkImageType type;
    VkFormat format;
    VkExtent3D extent;
    VkImageTiling tiling;
    VkImageUsageFlags usage;
};

VkFormatProperties getImageFormatProperties(VkPhysicalDevice ph_dev, VkFormat format);

VkFormat findSupportedImageFormat(VkPhysicalDevice ph_dev, std::span<VkFormat> candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features
);

VkImage createImage(VkDevice device, VkPhysicalDevice ph_dev, ImageParams params);

uint32_t findMemoryType(VkPhysicalDeviceMemoryProperties available,
    uint32_t type_filter, VkMemoryPropertyFlags requested
);

VkDeviceMemory allocateMemory(VkDevice device, VkPhysicalDeviceMemoryProperties properties,
    VkMemoryRequirements requirements, VkMemoryPropertyFlags flags, const void *export_info=nullptr
);

VkCommandPool createCommandPool(VkDevice device,
    uint32_t queue_idx, VkCommandPoolCreateFlags flags
);

std::vector<VkCommandBuffer> createCommandBuffers(VkDevice device,
    VkCommandPool command_pool, uint32_t buffer_count
);

VkDescriptorPool createDescriptorPool(VkDevice device,
    std::span<VkDescriptorPoolSize> pool_sizes
);

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device,
    std::span<VkDescriptorSetLayoutBinding> bindings
);

std::vector<VkDescriptorSet> createDescriptorSets(VkDevice device,
    VkDescriptorPool pool, VkDescriptorSetLayout layout, uint32_t set_count
);

VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptor_layout);

VkSampler createSampler(VkDevice device, VkFilter filter, bool enable_anisotropy);

VkFence createFence(VkDevice device, VkFenceCreateFlags flags);

VkSemaphore createSemaphore(VkDevice device, const void *extensions = nullptr);

VkQueryPool createQueryPool(VkDevice device, uint32_t query_count);

} // namespace mimir