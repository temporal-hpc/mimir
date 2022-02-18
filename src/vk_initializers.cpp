#include "internal/vk_initializers.hpp"

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

VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags)
{
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;
  info.pInheritanceInfo = nullptr;
  info.flags = flags;
  return info;
}

VkFramebufferCreateInfo framebufferCreateInfo(VkRenderPass pass, VkExtent2D extent)
{
  VkFramebufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.renderPass      = pass;
  info.attachmentCount = 1;
  info.width           = extent.width;
  info.height          = extent.height;
  info.layers          = 1;
  return info;
}

VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags)
{
  VkFenceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  return info;
}

VkSemaphoreCreateInfo semaphoreCreateInfo(VkSemaphoreCreateFlags flags)
{
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags;
  return info;
}

VkSubmitInfo submitInfo(VkCommandBuffer *cmd)
{
  VkSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.pNext = nullptr;
  info.waitSemaphoreCount   = 0;
  info.pWaitSemaphores      = nullptr;
  info.pWaitDstStageMask    = nullptr;
  info.commandBufferCount   = 1;
  info.pCommandBuffers      = cmd;
  info.signalSemaphoreCount = 0;
  info.pSignalSemaphores    = nullptr;
  return info;
}

VkPresentInfoKHR presentInfo()
{
  VkPresentInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.pNext = nullptr;
  info.swapchainCount     = 0;
  info.pSwapchains        = nullptr;
  info.waitSemaphoreCount = 0;
  info.pWaitSemaphores    = nullptr;
  info.pImageIndices      = nullptr;
  return info;
}

VkRenderPassBeginInfo renderPassBeginInfo(VkRenderPass pass,
  VkExtent2D win_extent, VkFramebuffer framebuffer)
{
  VkRenderPassBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  info.pNext = nullptr;
  info.renderPass        = pass;
  info.renderArea.offset = {0, 0};
  info.renderArea.extent = win_extent;
  info.clearValueCount   = 1;
  info.pClearValues      = nullptr;
  info.framebuffer       = framebuffer;
  return info;
}

VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
  VkShaderStageFlagBits stage, VkShaderModule module)
{
  VkPipelineShaderStageCreateInfo info{};
  info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext  = nullptr;
  info.stage  = stage;  // Shader stage
  info.module = module; // Module containing code for this shader stage
  info.pName  = "main"; // Shader entry point
  info.pSpecializationInfo = nullptr; // specify values for shader constants
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

VkImageCreateInfo imageCreateInfo(VkFormat format, VkImageUsageFlags flags,
  uint32_t width, uint32_t height)
{
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = nullptr;
  info.imageType     = VK_IMAGE_TYPE_2D;
  info.format        = format;
  info.extent.width  = width;
  info.extent.height = height;
  info.extent.depth  = 1;
  info.mipLevels     = 1;
  info.arrayLayers   = 1;
  info.samples       = VK_SAMPLE_COUNT_1_BIT;
  info.tiling        = VK_IMAGE_TILING_OPTIMAL;
  info.usage         = flags;
  return info;
}

VkImageViewCreateInfo imageViewCreateInfo(VkFormat format, VkImage image,
  VkImageAspectFlags flags)
{
  VkImageViewCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.pNext = nullptr;
  // Treat image as 1D/2D/3D texture or as a cube map
  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.image    = image;
  info.format   = format;
  // Describe image purpose and which part of it should be accesssed
  info.subresourceRange.baseMipLevel   = 0;
  info.subresourceRange.levelCount     = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount     = 1;
  info.subresourceRange.aspectMask     = flags;
  // Default mapping of all color channels
  info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  return info;
}

VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo(
  bool depth_test, bool depth_write, VkCompareOp compare_op)
{
  VkPipelineDepthStencilStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.depthTestEnable       = depth_test ? VK_TRUE : VK_FALSE;
  info.depthWriteEnable      = depth_write ? VK_TRUE : VK_FALSE;
  info.depthCompareOp        = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS;
  info.depthBoundsTestEnable = VK_FALSE;
  info.minDepthBounds        = 0.f;
  info.maxDepthBounds        = 1.f;
  info.stencilTestEnable     = VK_FALSE;
  return info;
}

VkDescriptorSetLayoutBinding descriptorLayoutBinding(
  uint32_t binding, VkDescriptorType type, VkShaderStageFlags flags)
{
  VkDescriptorSetLayoutBinding bind{};
  bind.binding            = binding;
  bind.descriptorCount    = 1;
  bind.descriptorType     = type;
  bind.pImmutableSamplers = nullptr;
  bind.stageFlags         = flags;
  return bind;
}

VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorType type,
  VkDescriptorSet dst_set, VkDescriptorBufferInfo *buffer_info, uint32_t binding)
{
  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext = nullptr;
  write.dstBinding      = binding;
  write.dstSet          = dst_set;
  write.descriptorCount = 1;
  write.descriptorType  = type;
  write.pBufferInfo     = buffer_info;
  return write;
}

VkWriteDescriptorSet writeDescriptorImage(VkDescriptorType type,
  VkDescriptorSet dst_set, VkDescriptorImageInfo *img_info, uint32_t binding)
{
  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext = nullptr;
  write.dstBinding      = binding;
  write.dstSet          = dst_set;
  write.descriptorCount = 1;
  write.descriptorType  = type;
  write.pImageInfo      = img_info;
  return write;
}

VkSamplerCreateInfo samplerCreateInfo(VkFilter filters, VkSamplerAddressMode mode)
{
  VkSamplerCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.pNext = nullptr;
  info.magFilter = filters;
  info.minFilter = filters;
  info.addressModeU = mode;
  info.addressModeV = mode;
  info.addressModeW = mode;
  return info;
}

} // namespace vkinit
