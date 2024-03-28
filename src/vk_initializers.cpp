#include "internal/vk_initializers.hpp"

// Convenience functions for building Vulkan info structures with default and/or
// reasonable values. It also helps look the code using these a little more tidy

namespace vkinit
{

VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags)
{
    VkCommandBufferBeginInfo info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .pInheritanceInfo = nullptr,
    };
    return info;
}

VkSubmitInfo submitInfo(VkCommandBuffer *cmd, std::span<VkSemaphore> waits,
    std::span<VkPipelineStageFlags> stages, std::span<VkSemaphore> signals,
    const void *timeline_info)
{
    VkSubmitInfo info{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = timeline_info,
        .waitSemaphoreCount   = (uint32_t)waits.size(),
        .pWaitSemaphores      = waits.data(),
        .pWaitDstStageMask    = stages.data(),
        .commandBufferCount   = 1,
        .pCommandBuffers      = cmd,
        .signalSemaphoreCount = (uint32_t)signals.size(),
        .pSignalSemaphores    = signals.data(),
    };
    return info;
}

VkImageCreateInfo imageCreateInfo(VkImageType type,
    VkFormat format, VkExtent3D extent, VkImageUsageFlags usage)
{
    VkImageCreateInfo info{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .flags       = 0,
        .imageType   = type,
        .format      = format,
        .extent      = extent,
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    return info;
}

VkImageViewCreateInfo imageViewCreateInfo(VkImage image,
    VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect_mask)
{
    VkImageViewCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = image,
        .viewType = view_type, // 1D/2D/3D texture, cubemap or array
        .format   = format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = aspect_mask,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    return info;
}

VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo *buffer_info)
{
    VkWriteDescriptorSet write{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext            = nullptr,
        .dstSet           = dst_set,
        .dstBinding       = binding,
        .dstArrayElement  = 0,
        .descriptorCount  = 1,
        .descriptorType   = type,
        .pImageInfo       = nullptr,
        .pBufferInfo      = buffer_info,
        .pTexelBufferView = nullptr,
    };
    return write;
}

VkWriteDescriptorSet writeDescriptorImage(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorImageInfo *img_info)
{
    VkWriteDescriptorSet write{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext            = nullptr,
        .dstSet           = dst_set,
        .dstBinding       = binding,
        .dstArrayElement  = 0,
        .descriptorCount  = 1,
        .descriptorType   = type,
        .pImageInfo       = img_info,
        .pBufferInfo      = nullptr,
        .pTexelBufferView = nullptr,
    };
    return write;
}

} // namespace vkinit
