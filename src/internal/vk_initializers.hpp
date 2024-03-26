#pragma once

#include <vulkan/vulkan.h>

#include <span> // std::span

namespace vkinit
{

constexpr uint32_t toInt32(long unsigned int size)
{
    return static_cast<uint32_t>(size);
}

// Command-related functions
VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags = 0);
VkSubmitInfo submitInfo(VkCommandBuffer *cmd, std::span<VkSemaphore> waits = {},
    std::span<VkPipelineStageFlags> stages = {}, std::span<VkSemaphore> signals = {},
    const void *timeline_info = nullptr
);

// Synchronization-related functions
VkSemaphoreCreateInfo semaphoreCreateInfo(const void *extensions = nullptr);

// Buffer/Image functions
VkBufferCreateInfo bufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage);
VkImageCreateInfo imageCreateInfo(VkImageType type,
    VkFormat format, VkExtent3D extent, VkImageUsageFlags usage
);
VkImageViewCreateInfo imageViewCreateInfo(VkImage image,
    VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect_mask
);

// Descriptor-related functions
VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo *buffer_info
);
VkWriteDescriptorSet writeDescriptorImage(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorImageInfo *img_info
);

} // namespace vkinit
