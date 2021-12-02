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

} // namespace vkinit
