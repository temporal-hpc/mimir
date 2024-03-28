#pragma once

#include <vulkan/vulkan.h>

#include <span> // std::span

namespace vkinit
{

// Command-related functions
VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags = 0);
VkSubmitInfo submitInfo(VkCommandBuffer *cmd, std::span<VkSemaphore> waits = {},
    std::span<VkPipelineStageFlags> stages = {}, std::span<VkSemaphore> signals = {},
    const void *timeline_info = nullptr
);

// Buffer/Image functions
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
