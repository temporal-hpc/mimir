#include "mimir/engine/vk_swapchain.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <limits> // std::numeric_limits

#include <mimir/validation.hpp>
#include "internal/vk_properties.hpp"

namespace mimir
{

VkPresentModeKHR getDesiredPresentMode(PresentOptions opts)
{
    switch (opts)
    {
        case PresentOptions::Immediate: return VK_PRESENT_MODE_IMMEDIATE_KHR;
        case PresentOptions::VSync: return VK_PRESENT_MODE_FIFO_KHR;
        case PresentOptions::TripleBuffering: return VK_PRESENT_MODE_MAILBOX_KHR;
        default: return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
}

VulkanSwapchain::~VulkanSwapchain()
{
    cleanup();
    main_deletors.flush();
}

void VulkanSwapchain::cleanup()
{
    aux_deletors.flush();
}

void VulkanSwapchain::initSurface(VkInstance instance, GLFWwindow *window)
{
    validation::checkVulkan(
        glfwCreateWindowSurface(instance, window, nullptr, &surface)
    );
    main_deletors.add([=,this](){
        vkDestroySurfaceKHR(instance, surface, nullptr);
    });
}

void VulkanSwapchain::create(uint32_t& width, uint32_t& height, PresentOptions opts,
    std::vector<uint32_t> queue_indices, VkPhysicalDevice physical_device, VkDevice device)
{
    VkSurfaceCapabilitiesKHR surf_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surf_caps);
    if (surf_caps.currentExtent.width == std::numeric_limits<uint32_t>::max())
    {
        extent.width = width;
        extent.height = height;
    }
    else
    {
        extent = surf_caps.currentExtent;
        width = extent.width;
        height = extent.height;
    }

    uint32_t mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &mode_count, nullptr
    );
    std::vector<VkPresentModeKHR> present_modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &mode_count, present_modes.data()
    );

    VkPresentModeKHR present_mode = getDesiredPresentMode(opts);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, surface, &format_count, nullptr
    );
    std::vector<VkSurfaceFormatKHR> surface_formats(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, surface, &format_count, surface_formats.data()
    );
    for (const auto& surf_format : surface_formats)
    {
        if (surf_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            surf_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            color_format = surf_format.format;
            color_space = surf_format.colorSpace;
        }
    }

    image_count = surf_caps.minImageCount + 1;
    const auto max_image_count = surf_caps.maxImageCount;
    if (max_image_count > 0 && image_count > max_image_count)
    {
        image_count = max_image_count;
    }

    // TODO: Delete old_swapchain after image_count frames have passed
    auto old_swapchain = VK_NULL_HANDLE; //swapchain;

    VkSwapchainCreateInfoKHR create_info{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext            = nullptr,
        .flags            = 0,
        .surface          = surface,
        .minImageCount    = image_count,
        .imageFormat      = color_format,
        .imageColorSpace  = color_space,
        .imageExtent      = extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .preTransform     = surf_caps.currentTransform,
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode      = present_mode,
        .clipped          = VK_TRUE,
        .oldSwapchain     = old_swapchain,
    };

    if (queue_indices[0] != queue_indices[1])
    {
        create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = queue_indices.size();
        create_info.pQueueFamilyIndices   = queue_indices.data();
    }
    validation::checkVulkan(vkCreateSwapchainKHR(
        device, &create_info, nullptr, &swapchain)
    );
    aux_deletors.add([=,this](){
        //printf("destroying swapchain\n");
        vkDestroySwapchainKHR(device, swapchain, nullptr);
    });

    vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
}

std::vector<VkImage> VulkanSwapchain::createImages(VkDevice device)
{
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
    std::vector<VkImage> images(image_count);
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, images.data());
    return images;
}

} // namespace mimir