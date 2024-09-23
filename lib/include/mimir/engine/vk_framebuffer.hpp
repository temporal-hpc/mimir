#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

namespace mimir
{

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

    void create(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
        VkImageView depth_view
    );
    uint32_t addAttachment(VkDevice device, VkImage image, VkFormat format);
};

} // namespace mimir