#include "vk_initializers.hpp"

namespace vkinit
{

VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queue_family_idx,
  VkCommandPoolCreateFlags flags)
{
  VkCommandPoolCreateInfo info{};
  info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext            = nullptr;
  info.queueFamilyIndex = queue_family_idx;
  info.flags            = flags;
  return info;
}

VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool,
  uint32_t count, VkCommandBufferLevel level)
{
  VkCommandBufferAllocateInfo info{};
  info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext              = nullptr;
  info.commandPool        = pool;
  info.commandBufferCount = count;
  info.level              = level;
  return info;
}

VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
  VkShaderStageFlagBits stage, VkShaderModule module)
{
  VkPipelineShaderStageCreateInfo info{};
  info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext  = nullptr;
  info.stage  = stage;
  info.module = module;
  info.pName  = "main";
  // Used to specify values for shader constants
  info.pSpecializationInfo = nullptr;
  return info;
}

VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo(
  const std::vector<VkVertexInputBindingDescription>& bindings,
  const std::vector<VkVertexInputAttributeDescription>& attributes)
{
  VkPipelineVertexInputStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.vertexBindingDescriptionCount   = bindings.size();
  info.pVertexBindingDescriptions      = bindings.empty()? nullptr : bindings.data();
  info.vertexAttributeDescriptionCount = attributes.size();
  info.pVertexAttributeDescriptions    = attributes.empty()? nullptr : attributes.data();
  return info;
}

VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo(
  VkPrimitiveTopology topology)
{
  VkPipelineInputAssemblyStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.topology               = topology;
  info.primitiveRestartEnable = VK_FALSE;
  return info;
}

VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode mode)
{
  VkPipelineRasterizationStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.depthClampEnable        = VK_FALSE;
  info.rasterizerDiscardEnable = VK_FALSE;
  info.polygonMode             = mode;
  info.lineWidth               = 1.f;
  info.cullMode                = VK_CULL_MODE_NONE;
  info.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  info.depthBiasEnable         = VK_FALSE;
  info.depthBiasConstantFactor = 0.f;
  info.depthBiasClamp          = 0.f;
  info.depthBiasSlopeFactor    = 0.f;
  return info;
}

VkPipelineMultisampleStateCreateInfo multisamplingStateCreateInfo()
{
  VkPipelineMultisampleStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.sampleShadingEnable   = VK_FALSE;
  info.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
  info.minSampleShading      = 1.f;
  info.pSampleMask           = nullptr;
  info.alphaToCoverageEnable = VK_FALSE;
  info.alphaToOneEnable      = VK_FALSE;
  return info;
}

VkPipelineColorBlendAttachmentState colorBlendAttachmentState()
{
  VkPipelineColorBlendAttachmentState attachment{};
  attachment.colorWriteMask      =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  attachment.blendEnable         = VK_FALSE;
  attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  attachment.colorBlendOp        = VK_BLEND_OP_ADD;
  attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  attachment.alphaBlendOp        = VK_BLEND_OP_ADD;
  return attachment;
}

VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
  const std::vector<VkDescriptorSetLayout>& layouts)
{
  VkPipelineLayoutCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;
  info.flags                  = 0;
  info.setLayoutCount         = layouts.size();
  info.pSetLayouts            = layouts.data();
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges    = nullptr;
  return info;
}

} // namespace vkinit
