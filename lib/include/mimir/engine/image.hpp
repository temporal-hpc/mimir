#pragma once

#include <vulkan/vulkan.h>

#include <span> // std::span

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

VkFormatProperties getFormatProperties(VkPhysicalDevice ph_dev, VkFormat format);

VkFormat findSupportedImageFormat(VkPhysicalDevice ph_dev, std::span<VkFormat> candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features
);

VkImage createImage(VkDevice device, VkPhysicalDevice ph_dev, ImageParams params);

} // namespace mimir