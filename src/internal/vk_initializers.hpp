#pragma once

#include <vulkan/vulkan.h>

#include <vector> // TODO: Use span

namespace vkinit
{

VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queue_family_idx,
  VkCommandPoolCreateFlags flags = 0
);

VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool,
  uint32_t count = 1, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY
);

VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags = 0);

VkFramebufferCreateInfo framebufferCreateInfo(VkRenderPass pass, VkExtent2D extent);

VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags = 0);

VkSemaphoreCreateInfo semaphoreCreateInfo(VkSemaphoreCreateFlags flags = 0);

VkSubmitInfo submitInfo(VkCommandBuffer *cmd);

VkPresentInfoKHR presentInfo();

VkRenderPassBeginInfo renderPassBeginInfo(VkRenderPass pass,
  VkExtent2D win_extent, VkFramebuffer framebuffer
);

VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
  VkShaderStageFlagBits stage, VkShaderModule module
);

VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo(
  const std::vector<VkVertexInputBindingDescription>& bindings = {},
  const std::vector<VkVertexInputAttributeDescription>& attributes = {}
);

VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo(VkPrimitiveTopology topology);

VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode mode);

VkPipelineMultisampleStateCreateInfo multisamplingStateCreateInfo();

VkPipelineColorBlendAttachmentState colorBlendAttachmentState();

VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
  const std::vector<VkDescriptorSetLayout>& layouts
);

VkImageCreateInfo imageCreateInfo(VkFormat format,
  VkImageUsageFlags flags, uint32_t width, uint32_t height
);

VkImageViewCreateInfo imageViewCreateInfo(
  VkFormat format, VkImage image, VkImageAspectFlags flags
);

VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo(
  bool depth_test, bool depth_write, VkCompareOp compare_op
);

VkDescriptorSetLayoutBinding descriptorLayoutBinding(
  uint32_t binding, VkDescriptorType type, VkShaderStageFlags flags
);

VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorType type,
  VkDescriptorSet dst_set, VkDescriptorBufferInfo *buffer_info, uint32_t binding
);

VkWriteDescriptorSet writeDescriptorImage(VkDescriptorType type,
  VkDescriptorSet dst_set, VkDescriptorImageInfo *img_info, uint32_t binding
);

VkSamplerCreateInfo samplerCreateInfo(VkFilter filters,
  VkSamplerAddressMode mode = VK_SAMPLER_ADDRESS_MODE_REPEAT
);

} // namespace vkinit
