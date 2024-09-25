#include <mimir/engine/image.hpp>

#include <spdlog/spdlog.h>
#include "internal/validation.hpp"

namespace mimir
{

VkFormatProperties getFormatProperties(VkPhysicalDevice ph_dev, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(ph_dev, format, &properties);
    return properties;
}

VkFormat findSupportedImageFormat(VkPhysicalDevice ph_dev, std::span<VkFormat> candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (auto format : candidates)
    {
        auto props = getFormatProperties(ph_dev, format);
        switch (tiling)
        {
            case VK_IMAGE_TILING_LINEAR:
            {
                if ((props.linearTilingFeatures & features) == features) return format;
                break;
            }
            case VK_IMAGE_TILING_OPTIMAL:
            {
                if ((props.optimalTilingFeatures & features) == features) return format;
                break;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkImage createImage(VkDevice device, VkPhysicalDevice ph_dev, ImageParams params)
{
    // TODO: Also check if texture is within bounds
    // auto max_dim = getMaxImageDimension(params.layout);
    // if (extent.width >= max_dim || extent.height >= max_dim || extent.height >= max_dim)
    // {
    //     spdlog::error("Requested image dimensions are larger than maximum");
    // }

    // Check that a Vulkan image handle can be created with the supplied parameters
    VkPhysicalDeviceImageFormatInfo2 format_info{
        .sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext  = nullptr,
        .format = params.format,
        .type   = params.type,
        .tiling = params.tiling,
        .usage  = params.usage,
        .flags  = 0
    };
    VkImageFormatProperties2 format_props{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = nullptr,
        .imageFormatProperties = {} // To be filled in function below
    };
    validation::checkVulkan(vkGetPhysicalDeviceImageFormatProperties2(
        ph_dev, &format_info, &format_props
    ));

    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkImageCreateInfo info{
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = &extmem_info,
        .flags                 = 0,
        .imageType             = params.type,
        .format                = params.format,
        .extent                = params.extent,
        .mipLevels             = 1,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,
        .tiling                = params.tiling,
        .usage                 = params.usage,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(device, &info, nullptr, &image));
    return image;
}

} // namespace mimir