#include "internal/vk_initializers.hpp"

// Convenience functions for building Vulkan info structures with default and/or 
// reasonable values. It also helps look the code using these a little more tidy

namespace vkinit
{

VkCommandPoolCreateInfo commandPoolCreateInfo(VkCommandPoolCreateFlags flags,
  uint32_t queue_family_idx)
{
  VkCommandPoolCreateInfo info{};
  info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext            = nullptr;
  info.flags            = flags;
  info.queueFamilyIndex = queue_family_idx;
  return info;
}

VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool,
  VkCommandBufferLevel level, uint32_t count)
{
  VkCommandBufferAllocateInfo info{};
  info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext              = nullptr;
  info.commandPool        = pool;
  info.level              = level;
  info.commandBufferCount = count;
  return info;
}

VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags)
{
  VkCommandBufferBeginInfo info{};
  info.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext            = nullptr;
  info.flags            = flags;
  info.pInheritanceInfo = nullptr;
  return info;
}

VkFramebufferCreateInfo framebufferCreateInfo(VkRenderPass pass, VkExtent2D extent)
{
  VkFramebufferCreateInfo info{};
  info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  info.pNext           = nullptr;
  info.flags           = 0; // Can be VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT
  info.renderPass      = pass;
  info.attachmentCount = 0;
  info.pAttachments    = nullptr;
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
  info.flags = flags; // Can be VK_FENCE_CREATE_SIGNALED_BIT
  return info;
}

VkSemaphoreCreateInfo semaphoreCreateInfo(VkSemaphoreCreateFlags flags)
{
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = flags; // Unused
  return info;
}

VkSubmitInfo submitInfo(VkCommandBuffer *cmd)
{
  VkSubmitInfo info{};
  info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.pNext                = nullptr;
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
  info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.pNext              = nullptr;
  info.waitSemaphoreCount = 0;
  info.pWaitSemaphores    = nullptr;
  info.swapchainCount     = 0;
  info.pSwapchains        = nullptr;
  info.pImageIndices      = nullptr;
  info.pResults           = nullptr;
  return info;
}

VkRenderPassBeginInfo renderPassBeginInfo(VkRenderPass pass,
  VkFramebuffer framebuffer, VkExtent2D win_extent)
{
  VkRenderPassBeginInfo info{};
  info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  info.pNext           = nullptr;
  info.renderPass      = pass;
  info.framebuffer     = framebuffer;
  info.renderArea      = { {0, 0}, win_extent };
  info.clearValueCount = 1;
  info.pClearValues    = nullptr;
  return info;
}

VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
  VkShaderStageFlagBits stage, VkShaderModule module)
{
  VkPipelineShaderStageCreateInfo info{};
  info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext  = nullptr;
  info.flags  = 0;
  info.stage  = stage;  // Shader stage
  info.module = module; // Module containing code for this shader stage
  info.pName  = "main"; // Shader entry point
  info.pSpecializationInfo = nullptr; // specify values for shader constants
  return info;
}

VkVertexInputBindingDescription vertexBindingDescription(
  uint32_t binding, uint32_t stride, VkVertexInputRate rate)
{
  VkVertexInputBindingDescription desc{};
  desc.binding   = binding;
  desc.stride    = stride;
  desc.inputRate = rate;
  return desc;
}

VkVertexInputAttributeDescription vertexAttributeDescription(
  uint32_t binding, uint32_t location, VkFormat format, uint32_t offset)
{
  VkVertexInputAttributeDescription desc{};
  desc.binding  = binding;
  desc.location = location;
  desc.format   = format;
  desc.offset   = offset;
  return desc;
}

VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo(
  const std::vector<VkVertexInputBindingDescription>& bindings,
  const std::vector<VkVertexInputAttributeDescription>& attributes)
{
  VkPipelineVertexInputStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0; // Currently unused
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
  info.flags                  = 0; // Currently unused
  info.topology               = topology;
  info.primitiveRestartEnable = VK_FALSE;
  return info;
}

VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode mode)
{
  VkPipelineRasterizationStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags                   = 0; // Currently unused
  info.depthClampEnable        = VK_FALSE;
  info.rasterizerDiscardEnable = VK_FALSE;
  info.polygonMode             = mode;
  info.cullMode                = VK_CULL_MODE_NONE;
  info.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  info.depthBiasEnable         = VK_FALSE;
  info.depthBiasConstantFactor = 0.f;
  info.depthBiasClamp          = 0.f;
  info.depthBiasSlopeFactor    = 0.f;
  info.lineWidth               = 1.f;
  return info;
}

VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo()
{
  VkPipelineMultisampleStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags                 = 0; // Currently unused
  info.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
  info.sampleShadingEnable   = VK_FALSE;
  info.minSampleShading      = 1.f;
  info.pSampleMask           = nullptr;
  info.alphaToCoverageEnable = VK_FALSE;
  info.alphaToOneEnable      = VK_FALSE;
  return info;
}

VkPipelineColorBlendAttachmentState colorBlendAttachmentState()
{
  VkPipelineColorBlendAttachmentState attachment{};
  attachment.blendEnable         = VK_TRUE;
  attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  attachment.colorBlendOp        = VK_BLEND_OP_ADD;
  attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  attachment.alphaBlendOp        = VK_BLEND_OP_SUBTRACT;
  attachment.colorWriteMask      =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  return attachment;
}

VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(
  const std::vector<VkDescriptorSetLayout>& layouts)
{
  VkPipelineLayoutCreateInfo info{};
  info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext                  = nullptr;
  info.flags                  = 0; // Currently unused
  info.setLayoutCount         = layouts.size();
  info.pSetLayouts            = layouts.data();
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges    = nullptr;
  return info;
}

VkImageCreateInfo imageCreateInfo(VkImageType type,
  VkFormat format, VkExtent3D extent, VkImageUsageFlags usage)
{
  VkImageCreateInfo info{};
  info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext       = nullptr;
  info.flags       = 0;
  info.imageType   = type;
  info.format      = format;
  info.extent      = extent;
  info.mipLevels   = 1;
  info.arrayLayers = 1;
  info.samples     = VK_SAMPLE_COUNT_1_BIT;
  info.tiling      = VK_IMAGE_TILING_OPTIMAL;
  info.usage       = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.queueFamilyIndexCount = 0;
  info.pQueueFamilyIndices   = nullptr;
  info.initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED;
  return info;
}

VkImageViewCreateInfo imageViewCreateInfo(VkImage image,
  VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect_mask)
{
  VkImageViewCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.pNext    = nullptr;
  info.flags    = 0;
  info.image    = image;
  info.viewType = view_type; // 1D/2D/3D texture, cubemap or array
  info.format   = format;
  // Default mapping of all color channels
  info.components.r = VK_COMPONENT_SWIZZLE_R;
  info.components.g = VK_COMPONENT_SWIZZLE_G;
  info.components.b = VK_COMPONENT_SWIZZLE_B;
  info.components.a = VK_COMPONENT_SWIZZLE_A;
  // Describe image purpose and which part of it should be accesssed
  info.subresourceRange.aspectMask     = aspect_mask;
  info.subresourceRange.baseMipLevel   = 0;
  info.subresourceRange.levelCount     = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount     = 1;
  return info;
}

VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo(
  bool depth_test, bool depth_write, VkCompareOp compare_op)
{
  VkPipelineDepthStencilStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0; // For additional depth/stencil state info
  info.depthTestEnable       = depth_test ? VK_TRUE : VK_FALSE;
  info.depthWriteEnable      = depth_write ? VK_TRUE : VK_FALSE;
  info.depthCompareOp        = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS;
  info.depthBoundsTestEnable = VK_FALSE;
  info.stencilTestEnable     = VK_FALSE;
  info.front                 = {}; // TODO: Get default values for both of these
  info.back                  = {};
  info.minDepthBounds        = 0.f;
  info.maxDepthBounds        = 1.f;
  return info;
}

VkDescriptorSetLayoutBinding descriptorLayoutBinding(
  uint32_t binding, VkDescriptorType type, VkShaderStageFlags flags)
{
  VkDescriptorSetLayoutBinding bind{};
  bind.binding            = binding;
  bind.descriptorType     = type;
  bind.descriptorCount    = 1;
  bind.stageFlags         = flags;
  bind.pImmutableSamplers = nullptr;
  return bind;
}

VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorSet dst_set,
  uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo *buffer_info)
{
  VkWriteDescriptorSet write{};
  write.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext            = nullptr;
  write.dstSet           = dst_set;
  write.dstBinding       = binding;
  write.dstArrayElement  = 0;
  write.descriptorCount  = 1;
  write.descriptorType   = type;
  write.pImageInfo       = nullptr;
  write.pBufferInfo      = buffer_info;
  write.pTexelBufferView = nullptr;
  return write;
}

VkWriteDescriptorSet writeDescriptorImage(VkDescriptorSet dst_set,
  uint32_t binding, VkDescriptorType type, VkDescriptorImageInfo *img_info)
{
  VkWriteDescriptorSet write{};
  write.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.pNext            = nullptr;
  write.dstSet           = dst_set;
  write.dstBinding       = binding;
  write.dstArrayElement  = 0;
  write.descriptorCount  = 1;
  write.descriptorType   = type;
  write.pImageInfo       = img_info;
  write.pBufferInfo      = nullptr;
  write.pTexelBufferView = nullptr;
  return write;
}

VkSamplerCreateInfo samplerCreateInfo(VkFilter filter, VkSamplerAddressMode mode)
{
  VkSamplerCreateInfo info{};
  info.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.pNext            = nullptr;
  info.flags            = 0;
  info.magFilter        = filter;
  info.minFilter        = filter;
  info.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU     = mode;
  info.addressModeV     = mode;
  info.addressModeW     = mode;
  info.mipLodBias       = 0.f;
  info.anisotropyEnable = VK_FALSE;
  info.maxAnisotropy    = 0.f;
  info.compareEnable    = VK_FALSE;
  info.compareOp        = VK_COMPARE_OP_NEVER;
  info.minLod           = 0.f;
  info.maxLod           = VK_LOD_CLAMP_NONE;
  info.borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  info.unnormalizedCoordinates = VK_FALSE;

  return info;
}

VkVertexInputAttributeDescription vertexDescription(
  uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
{
  VkVertexInputAttributeDescription desc{};
  desc.location = location;
  desc.binding  = binding;
  desc.format   = format;
  desc.offset   = offset;
  return desc;
}

VkBufferCreateInfo bufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage)
{
  VkBufferCreateInfo info{};
  info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext       = nullptr;
  info.flags       = 0; // Additional buffer parameters
  info.size        = size;
  info.usage       = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.queueFamilyIndexCount = 0;
  info.pQueueFamilyIndices   = nullptr;
  return info;
}

VkAttachmentDescription attachmentDescription(VkFormat format)
{
  VkAttachmentDescription desc{};
  desc.flags          = 0; // Can be VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
  desc.format         = format;
  desc.samples        = VK_SAMPLE_COUNT_1_BIT;
  desc.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  desc.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  desc.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  desc.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  desc.finalLayout    = VK_IMAGE_LAYOUT_UNDEFINED;
  return desc;
}

VkSubpassDescription subpassDescription(VkPipelineBindPoint bind_point)
{
  VkSubpassDescription desc{};
  desc.flags                   = 0; // Specify subpass usage
  desc.pipelineBindPoint       = bind_point;
  desc.inputAttachmentCount    = 0;
  desc.pInputAttachments       = nullptr;
  desc.colorAttachmentCount    = 0;
  desc.pColorAttachments       = nullptr;
  desc.pResolveAttachments     = nullptr;
  desc.pDepthStencilAttachment = nullptr;
  desc.pPreserveAttachments    = nullptr;
  return desc;
}

VkSubpassDependency subpassDependency()
{
  VkSubpassDependency dep{};
  dep.srcSubpass    = 0;
  dep.dstSubpass    = 0;
  dep.srcStageMask  = 0; // TODO: Change to VK_PIPELINE_STAGE_NONE in 1.3
  dep.dstStageMask  = 0;
  dep.srcAccessMask = 0; // TODO: Change to VK_ACCESS_NONE in 1.3
  dep.dstAccessMask = 0;
  return dep;
}

VkRenderPassCreateInfo renderPassCreateInfo()
{
  VkRenderPassCreateInfo info{};
  info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  info.pNext           = nullptr;
  info.flags           = 0; // Can be VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
  info.attachmentCount = 0;
  info.pAttachments    = nullptr;
  info.subpassCount    = 0;
  info.pSubpasses      = nullptr;
  info.dependencyCount = 0;
  info.pDependencies   = nullptr;
  return info;
}

VkPipelineViewportStateCreateInfo viewportCreateInfo()
{
  VkPipelineViewportStateCreateInfo info{};
  info.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  info.pNext         = nullptr;
  info.flags         = 0; // Unused
  info.viewportCount = 0;
  info.pViewports    = nullptr;
  info.scissorCount  = 0;
  info.pScissors     = nullptr;
  return info;
}

VkPipelineColorBlendStateCreateInfo colorBlendInfo()
{
  VkPipelineColorBlendStateCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  info.pNext             = nullptr;
  info.flags             = 0; // Can be VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM
  info.logicOpEnable     = VK_FALSE;
  info.logicOp           = VK_LOGIC_OP_NO_OP;
  info.attachmentCount   = 0;
  info.pAttachments      = nullptr;
  info.blendConstants[0] = 0.f;
  info.blendConstants[1] = 0.f;
  info.blendConstants[2] = 0.f;
  info.blendConstants[3] = 0.f;
  return info;
}

VkGraphicsPipelineCreateInfo pipelineCreateInfo(VkPipelineLayout layout,
  VkRenderPass render_pass)
{
  VkGraphicsPipelineCreateInfo info{};
  info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  info.pNext               = nullptr;
  info.flags               = 0; // Specify how the pipeline is created
  info.stageCount          = 0;
  info.pStages             = nullptr;
  info.pVertexInputState   = nullptr;
  info.pInputAssemblyState = nullptr;
  info.pTessellationState  = nullptr;
  info.pViewportState      = nullptr;
  info.pRasterizationState = nullptr;
  info.pMultisampleState   = nullptr;
  info.pDepthStencilState  = nullptr;
  info.pColorBlendState    = nullptr;
  info.pDynamicState       = nullptr;
  info.layout              = layout;
  info.renderPass          = render_pass;
  info.subpass             = 0;
  info.basePipelineHandle  = VK_NULL_HANDLE;
  info.basePipelineIndex   = -1;
  return info;
}

} // namespace vkinit
