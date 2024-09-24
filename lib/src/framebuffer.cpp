#include <mimir/engine/framebuffer.hpp>

#include "internal/validation.hpp"

namespace mimir
{

VkImageViewCreateInfo attachmentViewInfo(VkImage image, VkFormat format)
{
    return {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, // 1D/2D/3D texture, cubemap or array
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
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
}

Framebuffer Framebuffer::make(VkDevice device, VkPhysicalDevice ph_dev,
    Swapchain swapchain, VkRenderPass render_pass)
{
    auto images = swapchain.getImages(device);
    std::vector<VkImageView> views;
    views.reserve(images.size() + 1); // Color attachments + depth view

    for (const auto& image : images)
    {
        auto info = attachmentViewInfo(image, swapchain.format);
        VkImageView view = VK_NULL_HANDLE;
        validation::checkVulkan(vkCreateImageView(device, &info, nullptr, &view));
        views.push_back(view);
    }

    VkFramebufferCreateInfo info{
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT
        .renderPass      = render_pass,
        .attachmentCount = (uint32_t)views.size(),
        .pAttachments    = views.data(),
        .width           = swapchain.extent.width,
        .height          = swapchain.extent.height,
        .layers          = 1,
    };
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateFramebuffer(device, &info, nullptr, &framebuffer)
    );

    return {
        .handle = framebuffer,
    };
}

uint32_t VulkanFramebuffer::addAttachment(VkDevice device,
    VkImage image, VkFormat format)
{
    FramebufferAttachment attachment{
        .image  = image,
        .view   = VK_NULL_HANDLE,
        .format = format,
    };

    VkImageViewCreateInfo view_info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = attachment.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, // 1D/2D/3D texture, cubemap or array
        .format   = attachment.format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    validation::checkVulkan(
        vkCreateImageView(device, &view_info, nullptr, &attachment.view)
    );

    attachments.push_back(attachment);
    return 0;
}

void VulkanFramebuffer::create(VkDevice device,
  VkRenderPass render_pass, VkExtent2D extent, VkImageView depth_view)
{
    std::vector<VkImageView> attachment_views;
    for (auto attachment : attachments)
    {
        attachment_views.push_back(attachment.view);
    }
    attachment_views.push_back(depth_view);

    VkFramebufferCreateInfo info{
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT
        .renderPass      = render_pass,
        .attachmentCount = (uint32_t)attachment_views.size(),
        .pAttachments    = attachment_views.data(),
        .width           = extent.width,
        .height          = extent.height,
        .layers          = 1,
    };

    validation::checkVulkan(
        vkCreateFramebuffer(device, &info, nullptr, &framebuffer)
    );
}

} // namespace mimir