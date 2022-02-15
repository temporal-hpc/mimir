#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>

#include "cudaview/deletion_queue.hpp"

struct FramebufferAttachment
{
  VkImage image;
  VkImageView view;
  VkFormat format;
};

struct VulkanFramebuffer
{
  VkExtent2D extent;
  VkFramebuffer framebuffer;
  VkRenderPass render_pass;
  VkSampler sampler;
  std::vector<FramebufferAttachment> attachments;
  DeletionQueue deletors;

  ~VulkanFramebuffer();
  void createSampler(VkDevice device);
  void create(VkDevice device, VkRenderPass render_pass, VkExtent2D extent);
  uint32_t addAttachment(VkDevice device, VkImage image, VkFormat format);
};

VkRenderPass createRenderPass(VkDevice device, VkFormat color_format);
