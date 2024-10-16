#include "internal/swapchain.hpp"

#include "internal/validation.hpp"

namespace mimir
{

Swapchain Swapchain::make(VkDevice device, VkPhysicalDevice ph_dev, VkSurfaceKHR surf,
    int width, int height, VkPresentModeKHR mode, std::vector<uint32_t> queue_indices)
{
    Swapchain sc{
        .current     = VK_NULL_HANDLE,
        .old         = VK_NULL_HANDLE,
        .format      = VK_FORMAT_UNDEFINED,
        .extent      = { 0, 0 },
        .image_count = 0,
    };

    // Get a swapchain image count within surface supported limits
    VkSurfaceCapabilitiesKHR surf_caps;
    validation::checkVulkan(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        ph_dev, surf, &surf_caps)
    );
    sc.image_count = surf_caps.minImageCount + 1;
    auto max_image_count = surf_caps.maxImageCount;
    if (max_image_count > 0 && sc.image_count > max_image_count)
    {
        sc.image_count = max_image_count;
    }

    sc.extent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    // if (surf_caps.currentExtent.width == std::numeric_limits<uint32_t>::max())
    // {
    //     extent.width = width;
    //     extent.height = height;
    // }
    // else
    // {
    //     extent = surf_caps.currentExtent;
    //     width = extent.width;
    //     height = extent.height;
    // }

    // Retrieve list of surface formats supported by the current device
    uint32_t format_count;
    validation::checkVulkan(vkGetPhysicalDeviceSurfaceFormatsKHR(
        ph_dev, surf, &format_count, nullptr)
    );
    std::vector<VkSurfaceFormatKHR> surface_formats(format_count);
    validation::checkVulkan(vkGetPhysicalDeviceSurfaceFormatsKHR(
        ph_dev, surf, &format_count, surface_formats.data())
    );

    // Go through available image formats and select expected image format & color space
    VkColorSpaceKHR color_space;
    for (const auto& surf_format : surface_formats)
    {
        if (surf_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            surf_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            sc.format = surf_format.format;
            color_space = surf_format.colorSpace;
        }
    }

    auto sharing_mode = (queue_indices.size() > 1 && queue_indices[0] != queue_indices[1])?
        VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;

    // Create swapchain
    VkSwapchainCreateInfoKHR info{
        .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext                 = nullptr,
        .flags                 = 0,
        .surface               = surf,
        .minImageCount         = sc.image_count,
        .imageFormat           = sc.format,
        .imageColorSpace       = color_space,
        .imageExtent           = sc.extent,
        .imageArrayLayers      = 1,
        .imageUsage            = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode      = sharing_mode,
        .queueFamilyIndexCount = static_cast<uint32_t>(queue_indices.size()),
        .pQueueFamilyIndices   = queue_indices.data(),
        .preTransform          = surf_caps.currentTransform,
        .compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode           = mode,
        .clipped               = VK_TRUE,
        .oldSwapchain          = nullptr,
    };
    validation::checkVulkan(vkCreateSwapchainKHR(device, &info, nullptr, &sc.current));

    return sc;
}

std::vector<VkImage> Swapchain::getImages(VkDevice device)
{
    std::vector<VkImage> images(image_count);
    vkGetSwapchainImagesKHR(device, current, &image_count, images.data());
    return images;
}

} // namespace mimir