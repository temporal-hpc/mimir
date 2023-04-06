#pragma once

#include <vulkan/vulkan.hpp>

#include <vector> // std::vector

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
    void create(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
        VkImageView depth_view
    );
    uint32_t addAttachment(VkDevice device, VkImage image, VkFormat format);
};
