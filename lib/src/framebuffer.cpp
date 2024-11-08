#include "internal/framebuffer.hpp"

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

Framebuffer Framebuffer::make(VkDevice device, VkRenderPass render_pass,
    Swapchain swapchain, VkImageView depth_view)
{
    auto images = swapchain.getImages(device);

    Framebuffer fb{
        .handles =     {},
        .image_views = {},
    };
    fb.handles.reserve(images.size());
    fb.image_views.reserve(images.size());

    for (const auto& image : images)
    {
        auto info = attachmentViewInfo(image, swapchain.format);
        VkImageView color_view = VK_NULL_HANDLE;
        validation::checkVulkan(vkCreateImageView(device, &info, nullptr, &color_view));

        std::vector<VkImageView> attachments = { color_view, depth_view };
        VkFramebufferCreateInfo fb_info{
            .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext           = nullptr,
            .flags           = 0, // Can be VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT
            .renderPass      = render_pass,
            .attachmentCount = (uint32_t)attachments.size(),
            .pAttachments    = attachments.data(),
            .width           = swapchain.extent.width,
            .height          = swapchain.extent.height,
            .layers          = 1,
        };
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        validation::checkVulkan(vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer));

        fb.image_views.push_back(color_view);
        fb.handles.push_back(framebuffer);
    }
    return fb;
}

} // namespace mimir