#include "mimir/engine/vk_framebuffer.hpp"

#include <mimir/validation.hpp>
#include "internal/vk_initializers.hpp"

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
    FramebufferAttachment attachment;
    attachment.image  = image;
    attachment.format = format;

    auto view_info = vkinit::imageViewCreateInfo(attachment.image,
        VK_IMAGE_VIEW_TYPE_2D, attachment.format, VK_IMAGE_ASPECT_COLOR_BIT
    );
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

    auto fb_info = vkinit::framebufferCreateInfo(render_pass, extent);
    fb_info.pAttachments    = attachment_views.data();
    fb_info.attachmentCount = attachment_views.size();
    //printf("attachment size %lu\n", attachment_views.size());

    validation::checkVulkan(
        vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer)
    );
    deletors.add([=,this](){
        //printf("destroying framebuffer\n");
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    });
}

} // namespace mimir