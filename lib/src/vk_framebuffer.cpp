#include "mimir/engine/vk_framebuffer.hpp"

#include <mimir/validation.hpp>

namespace mimir
{

VulkanFramebuffer::~VulkanFramebuffer()
{
    //printf("framebuffer flush\n");
    deletors.flush();
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
    deletors.add([=](){
        //printf("destroying attachment imageview\n");
        vkDestroyImageView(device, attachment.view, nullptr);
    });

    attachments.push_back(attachment);
    //printf("Adding attachment\n");

    return 0;
}

void VulkanFramebuffer::create(VkDevice device,
  VkRenderPass render_pass, VkExtent2D extent, VkImageView depth_view)
{
    //printf("creating framebuffer\n");
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
    //printf("attachment size %lu\n", attachment_views.size());

    validation::checkVulkan(
        vkCreateFramebuffer(device, &info, nullptr, &framebuffer)
    );
    deletors.add([=,this](){
        //printf("destroying framebuffer\n");
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    });
}

} // namespace mimir