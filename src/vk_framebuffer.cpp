#include "cudaview/engine/vk_framebuffer.hpp"

#include "internal/validation.hpp"
#include "cudaview/engine/vk_initializers.hpp"

VulkanFramebuffer::~VulkanFramebuffer()
{
  deletors.flush();
  /*for (auto attachment : attachments)
  {
    vkDestroyImageView(device, attachment.view, nullptr);
  }
  vkDestroySampler(device, sampler, nullptr);
  vkDestroyRenderPass(device, render_pass, nullptr);
  vkDestroyFramebuffer(device, framebuffer, nullptr);*/
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

  auto view_info = vkinit::imageViewCreateInfo(
    attachment.format, attachment.image, VK_IMAGE_ASPECT_COLOR_BIT
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
  VkRenderPass render_pass, VkExtent2D extent)
{
  std::vector<VkImageView> attachment_views;
  for (auto attachment : attachments)
  {
    attachment_views.push_back(attachment.view);
  }

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

VkRenderPass createRenderPass(VkDevice device, VkFormat color_format)
{
  VkAttachmentDescription color_attachment{};
  color_attachment.format         = color_format;
  color_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &color_attachment_ref;

  // Specify memory and execution dependencies between subpasses
  VkSubpassDependency dependency{};
  dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass    = 0;
  dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo pass_info{};
  pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  pass_info.attachmentCount = 1;
  pass_info.pAttachments    = &color_attachment;
  pass_info.subpassCount    = 1;
  pass_info.pSubpasses      = &subpass;
  pass_info.dependencyCount = 1;
  pass_info.pDependencies   = &dependency;

  VkRenderPass render_pass;
  validation::checkVulkan(
    vkCreateRenderPass(device, &pass_info, nullptr, &render_pass)
  );
  return render_pass;
}
