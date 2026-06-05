#pragma once

#include <vulkan/vulkan.h>

#include "swapchain.hpp"
#include <vector> // std::vector
#include <span> // std::span

namespace mimir
{

struct Framebuffer
{
    std::vector<VkFramebuffer> handles;
    std::vector<VkImageView> image_views;

    static Framebuffer make(VkDevice device, VkRenderPass render_pass,
        Swapchain swapchain, VkImageView depth_view
    );

    // Builds framebuffers from an explicit set of color images (e.g. an offscreen target),
    // rather than from a swapchain. Used by headless/offscreen rendering.
    static Framebuffer make(VkDevice device, VkRenderPass render_pass,
        std::span<const VkImage> images, VkFormat format, VkExtent2D extent,
        VkImageView depth_view
    );
};

} // namespace mimir