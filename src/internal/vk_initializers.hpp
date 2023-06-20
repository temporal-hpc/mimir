#pragma once

#include <vulkan/vulkan.h>

#include <span> // std::span

namespace vkinit
{

// Command-related functions
VkCommandPoolCreateInfo commandPoolCreateInfo(VkCommandPoolCreateFlags flags,
    uint32_t queue_family_idx
);
VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, uint32_t count = 1
);
VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags = 0);
VkSubmitInfo submitInfo(VkCommandBuffer *cmd, std::span<VkSemaphore> waits = {},
    std::span<VkPipelineStageFlags> stages = {}, std::span<VkSemaphore> signals = {},
    const void *timeline_info = nullptr
);

// Synchronization-related functions
VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags = 0);
VkSemaphoreCreateInfo semaphoreCreateInfo(VkSemaphoreCreateFlags flags = 0);

// Presentation-related functions
VkPresentInfoKHR presentInfo(uint32_t *image_ids, VkSwapchainKHR *swapchain, VkSemaphore *semaphore);
VkRenderPassBeginInfo renderPassBeginInfo(VkRenderPass pass,
    VkFramebuffer framebuffer, VkExtent2D win_extent
);

// Pipeline-related functions
VkVertexInputBindingDescription vertexBindingDescription(
    uint32_t binding, uint32_t stride, VkVertexInputRate rate
);
VkVertexInputAttributeDescription vertexAttributeDescription(
    uint32_t binding, uint32_t location, VkFormat format, uint32_t offset
);
VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
    std::span<VkDescriptorSetLayout> layouts
);
VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule module
);
VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo(
    std::span<VkVertexInputBindingDescription> bindings = {},
    std::span<VkVertexInputAttributeDescription> attributes = {}
);
VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo(
    bool depth_test, bool depth_write, VkCompareOp compare_op
);
VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo(VkPrimitiveTopology topology);
VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode mode);
VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo();
VkPipelineColorBlendAttachmentState colorBlendAttachmentState();
VkPipelineColorBlendStateCreateInfo colorBlendInfo();
VkPipelineViewportStateCreateInfo viewportCreateInfo();
VkGraphicsPipelineCreateInfo pipelineCreateInfo(VkPipelineLayout layout,
    VkRenderPass render_pass
);

// Image-related functions
VkImageCreateInfo imageCreateInfo(VkImageType type,
    VkFormat format, VkExtent3D extent, VkImageUsageFlags usage
);
VkImageViewCreateInfo imageViewCreateInfo(VkImage image,
    VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect_mask
);

// Descriptor-related functions
VkDescriptorSetLayoutBinding descriptorLayoutBinding(
    uint32_t binding, VkDescriptorType type, VkShaderStageFlags flags
);
VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo *buffer_info
);
VkWriteDescriptorSet writeDescriptorImage(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorImageInfo *img_info
);

// Renderpass-related functions
VkAttachmentDescription attachmentDescription(VkFormat format);
VkSubpassDescription subpassDescription(VkPipelineBindPoint bind_point);
VkSubpassDependency subpassDependency();
VkRenderPassCreateInfo renderPassCreateInfo();
VkFramebufferCreateInfo framebufferCreateInfo(VkRenderPass pass, VkExtent2D extent);

// Other functions
VkSamplerCreateInfo samplerCreateInfo(VkFilter filter,
    VkSamplerAddressMode mode = VK_SAMPLER_ADDRESS_MODE_REPEAT
);
VkVertexInputAttributeDescription vertexDescription(
    uint32_t location, uint32_t binding, VkFormat format, uint32_t offset
);
VkBufferCreateInfo bufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage);

} // namespace vkinit
