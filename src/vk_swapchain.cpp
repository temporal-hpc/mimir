#include "cudaview/engine/vk_swapchain.hpp"

#include <limits> // std::numeric_limits

#include <cudaview/validation.hpp>
#include "internal/vk_initializers.hpp"
#include "internal/vk_properties.hpp"

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

void VulkanSwapchain::create(uint32_t& width, uint32_t& height,
    std::vector<uint32_t> queue_indices, VkPhysicalDevice physical_device, VkDevice device)
{
    VkSurfaceCapabilitiesKHR surf_caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surf_caps);
    if (surf_caps.currentExtent.width == std::numeric_limits<uint32_t>::max())
    {
        swapchain_extent.width = width;
        swapchain_extent.height = height;
    }
    else
    {
        swapchain_extent = surf_caps.currentExtent;
        width = swapchain_extent.width;
        height = swapchain_extent.height;
    }

    uint32_t mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &mode_count, nullptr
    );
    std::vector<VkPresentModeKHR> present_modes(mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &mode_count, present_modes.data()
    );
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (const auto& mode : present_modes)
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            present_mode = mode;
            break;
        }
        else if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            present_mode = mode;
        }
    }

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
        if (surf_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
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

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface          = surface;
    create_info.minImageCount    = image_count;
    create_info.imageFormat      = color_format;
    create_info.imageColorSpace  = color_space;
    create_info.imageExtent      = swapchain_extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (queue_indices[0] != queue_indices[1])
    {
        create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = queue_indices.size();
        create_info.pQueueFamilyIndices   = queue_indices.data();
    }
    else
    {
        create_info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
        create_info.queueFamilyIndexCount = 0;
        create_info.pQueueFamilyIndices   = nullptr;
    }
    create_info.preTransform     = surf_caps.currentTransform;
    create_info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode      = present_mode;
    create_info.clipped          = VK_TRUE;
    create_info.oldSwapchain     = old_swapchain;

    validation::checkVulkan(vkCreateSwapchainKHR(
        device, &create_info, nullptr, &swapchain)
    );
    aux_deletors.add([=,this](){
        printf("destroying swapchain\n");
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
