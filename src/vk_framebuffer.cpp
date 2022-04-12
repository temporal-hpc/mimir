#include "cudaview/engine/vk_framebuffer.hpp"

#include "internal/validation.hpp"
#include "internal/vk_initializers.hpp"

VulkanFramebuffer::~VulkanFramebuffer()
{
  deletors.flush();
}

void VulkanFramebuffer::createSampler(VkDevice device)
{
  auto sampler_info = vkinit::samplerCreateInfo(VK_FILTER_LINEAR);
  sampler_info.anisotropyEnable        = VK_TRUE;
  //sampler_info.maxAnisotropy           = max_anisotropy;
  sampler_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable           = VK_FALSE;
  sampler_info.compareOp               = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.mipLodBias              = 0.f;
  sampler_info.minLod                  = 0.f;
  sampler_info.maxLod                  = 0.f;

  validation::checkVulkan(
    vkCreateSampler(device, &sampler_info, nullptr, &sampler)
  );
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
  deletors.pushFunction([=](){
    vkDestroyImageView(device, attachment.view, nullptr);
  });

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

  auto fb_info = vkinit::framebufferCreateInfo(render_pass, extent);
  fb_info.pAttachments    = attachment_views.data();
  fb_info.attachmentCount = attachment_views.size();

  validation::checkVulkan(
    vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffer)
  );
  deletors.pushFunction([=](){
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  });
}
