#include "internal/vk_initializers.hpp"

// Convenience functions for building Vulkan info structures with default and/or
// reasonable values. It also helps look the code using these a little more tidy

namespace vkinit
{

VkCommandPoolCreateInfo commandPoolCreateInfo(VkCommandPoolCreateFlags flags,
    uint32_t queue_family_idx)
{
    VkCommandPoolCreateInfo info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .queueFamilyIndex = queue_family_idx,
    };
    return info;
}

VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool,
    VkCommandBufferLevel level, uint32_t count)
{
    VkCommandBufferAllocateInfo info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool,
        .level              = level,
        .commandBufferCount = count,
    };
    return info;
}

VkCommandBufferBeginInfo commandBufferBeginInfo(VkCommandBufferUsageFlags flags)
{
    VkCommandBufferBeginInfo info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .pInheritanceInfo = nullptr,
    };
    return info;
}

VkFramebufferCreateInfo framebufferCreateInfo(VkRenderPass pass, VkExtent2D extent)
{
    VkFramebufferCreateInfo info{
        .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT
        .renderPass      = pass,
        .attachmentCount = 0,
        .pAttachments    = nullptr,
        .width           = extent.width,
        .height          = extent.height,
        .layers          = 1,
    };
    return info;
}

VkFenceCreateInfo fenceCreateInfo(VkFenceCreateFlags flags)
{
    VkFenceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags, // Can be VK_FENCE_CREATE_SIGNALED_BIT
    };
    return info;
}

VkSemaphoreCreateInfo semaphoreCreateInfo(const void *extensions)
{
    VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = extensions,
        .flags = 0, // Unused
    };
    return info;
}

VkTimelineSemaphoreSubmitInfo timelineSubmitInfo(uint64_t *wait, uint64_t *signal)
{
    VkTimelineSemaphoreSubmitInfo info{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount   = 1,
        .pWaitSemaphoreValues      = wait,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues    = signal,
    };
    return info;
}

VkSubmitInfo submitInfo(VkCommandBuffer *cmd, std::span<VkSemaphore> waits,
    std::span<VkPipelineStageFlags> stages, std::span<VkSemaphore> signals,
    const void *timeline_info)
{
    VkSubmitInfo info{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = timeline_info,
        .waitSemaphoreCount   = toInt32(waits.size()),
        .pWaitSemaphores      = waits.data(),
        .pWaitDstStageMask    = stages.data(),
        .commandBufferCount   = 1,
        .pCommandBuffers      = cmd,
        .signalSemaphoreCount = toInt32(signals.size()),
        .pSignalSemaphores    = signals.data(),
    };
    return info;
}

VkPresentInfoKHR presentInfo(uint32_t *image_ids, VkSwapchainKHR *swapchain, VkSemaphore *semaphore)
{
    VkPresentInfoKHR info{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext              = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = semaphore,
        .swapchainCount     = 1,
        .pSwapchains        = swapchain,
        .pImageIndices      = image_ids,
        .pResults           = nullptr,
    };
    return info;
}

VkRenderPassBeginInfo renderPassBeginInfo(VkRenderPass pass,
    VkFramebuffer framebuffer, VkExtent2D win_extent)
{
    VkRenderPassBeginInfo info{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext           = nullptr,
        .renderPass      = pass,
        .framebuffer     = framebuffer,
        .renderArea      = { {0, 0}, win_extent },
        .clearValueCount = 1,
        .pClearValues    = nullptr,
    };
    return info;
}

VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule module)
{
    VkPipelineShaderStageCreateInfo info{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext  = nullptr,
        .flags  = 0,
        .stage  = stage,  // Shader stage
        .module = module, // Module containing code for this shader stage
        .pName  = "main", // Shader entry point
        .pSpecializationInfo = nullptr, // specify values for shader constants
    };
    return info;
}

VkVertexInputBindingDescription vertexBindingDescription(
    uint32_t binding, uint32_t stride, VkVertexInputRate rate)
{
    VkVertexInputBindingDescription desc{
        .binding   = binding,
        .stride    = stride,
        .inputRate = rate,
    };
    return desc;
}

VkVertexInputAttributeDescription vertexAttributeDescription(
    uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
{
    VkVertexInputAttributeDescription desc{
        .location = location,
        .binding  = binding,
        .format   = format,
        .offset   = offset,
    };
    return desc;
}

VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo(
    std::span<VkVertexInputBindingDescription> bindings,
    std::span<VkVertexInputAttributeDescription> attributes)
{
    VkPipelineVertexInputStateCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0, // Currently unused
        .vertexBindingDescriptionCount   = toInt32(bindings.size()),
        .pVertexBindingDescriptions      = bindings.empty()? nullptr : bindings.data(),
        .vertexAttributeDescriptionCount = toInt32(attributes.size()),
        .pVertexAttributeDescriptions    = attributes.empty()? nullptr : attributes.data(),
    };
    return info;
}

VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo(
    VkPrimitiveTopology topology)
{
    VkPipelineInputAssemblyStateCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0, // Currently unused
        .topology = topology,
        .primitiveRestartEnable = VK_FALSE,
    };
    return info;
}

VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode mode)
{
    VkPipelineRasterizationStateCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags                   = 0, // Currently unused
        .depthClampEnable        = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode             = mode,
        .cullMode                = VK_CULL_MODE_NONE,
        .frontFace               = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable         = VK_FALSE,
        .depthBiasConstantFactor = 0.f,
        .depthBiasClamp          = 0.f,
        .depthBiasSlopeFactor    = 0.f,
        .lineWidth               = 1.f,
    };
    return info;
}

VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo()
{
    VkPipelineMultisampleStateCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0, // Currently unused
        .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable   = VK_FALSE,
        .minSampleShading      = 1.f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable      = VK_FALSE,
    };
    return info;
}

VkPipelineColorBlendAttachmentState colorBlendAttachmentState()
{
    VkPipelineColorBlendAttachmentState attachment{
        .blendEnable         = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp        = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp        = VK_BLEND_OP_ADD,
        .colorWriteMask      =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    return attachment;
}

VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo(std::span<VkDescriptorSetLayout> layouts)
{
    VkPipelineLayoutCreateInfo info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0, // Currently unused
        .setLayoutCount         = toInt32(layouts.size()),
        .pSetLayouts            = layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges    = nullptr,
    };
    return info;
}

VkImageCreateInfo imageCreateInfo(VkImageType type,
    VkFormat format, VkExtent3D extent, VkImageUsageFlags usage)
{
    VkImageCreateInfo info{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .flags       = 0,
        .imageType   = type,
        .format      = format,
        .extent      = extent,
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    return info;
}

VkImageViewCreateInfo imageViewCreateInfo(VkImage image,
    VkImageViewType view_type, VkFormat format, VkImageAspectFlags aspect_mask)
{
    VkImageViewCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = image,
        .viewType = view_type, // 1D/2D/3D texture, cubemap or array
        .format   = format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,            
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = aspect_mask,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    return info;
}

VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo(
    bool depth_test, bool depth_write, VkCompareOp compare_op)
{
    VkPipelineDepthStencilStateCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0, // For additional depth/stencil state info
        .depthTestEnable       = depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable      = depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp        = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable     = VK_FALSE,
        .front                 = {}, // TODO: Get default values for both of these
        .back                  = {},
        .minDepthBounds        = 0.f,
        .maxDepthBounds        = 1.f,
    };
    return info;
}

VkDescriptorSetLayoutBinding descriptorLayoutBinding(
    uint32_t binding, VkDescriptorType type, VkShaderStageFlags flags)
{
    VkDescriptorSetLayoutBinding bind{
        .binding            = binding,
        .descriptorType     = type,
        .descriptorCount    = 1,
        .stageFlags         = flags,
        .pImmutableSamplers = nullptr,
    };
    return bind;
}

VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo *buffer_info)
{
    VkWriteDescriptorSet write{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext            = nullptr,
        .dstSet           = dst_set,
        .dstBinding       = binding,
        .dstArrayElement  = 0,
        .descriptorCount  = 1,
        .descriptorType   = type,
        .pImageInfo       = nullptr,
        .pBufferInfo      = buffer_info,
        .pTexelBufferView = nullptr,
    };
    return write;
}

VkWriteDescriptorSet writeDescriptorImage(VkDescriptorSet dst_set,
    uint32_t binding, VkDescriptorType type, VkDescriptorImageInfo *img_info)
{
    VkWriteDescriptorSet write{
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext            = nullptr,
        .dstSet           = dst_set,
        .dstBinding       = binding,
        .dstArrayElement  = 0,
        .descriptorCount  = 1,
        .descriptorType   = type,
        .pImageInfo       = img_info,
        .pBufferInfo      = nullptr,
        .pTexelBufferView = nullptr,
    };
    return write;
}

VkSamplerCreateInfo samplerCreateInfo(VkFilter filter, VkSamplerAddressMode mode)
{
    VkSamplerCreateInfo info{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .magFilter        = filter,
        .minFilter        = filter,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU     = mode,
        .addressModeV     = mode,
        .addressModeW     = mode,
        .mipLodBias       = 0.f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy    = 0.f,
        .compareEnable    = VK_FALSE,
        .compareOp        = VK_COMPARE_OP_NEVER,
        .minLod           = 0.f,
        .maxLod           = VK_LOD_CLAMP_NONE,
        .borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    return info;
}

VkVertexInputAttributeDescription vertexDescription(
    uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
{
    VkVertexInputAttributeDescription desc{
        .location = location,
        .binding  = binding,
        .format   = format,
        .offset   = offset,
    };
    return desc;
}

VkBufferCreateInfo bufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo info{
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .flags       = 0, // Additional buffer parameters
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };
    return info;
}

VkAttachmentDescription attachmentDescription(VkFormat format)
{
    VkAttachmentDescription desc{
        .flags          = 0, // Can be VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
        .format         = format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    return desc;
}

VkSubpassDescription subpassDescription(VkPipelineBindPoint bind_point)
{
    VkSubpassDescription desc{
        .flags                   = 0, // Specify subpass usage
        .pipelineBindPoint       = bind_point,
        .inputAttachmentCount    = 0,
        .pInputAttachments       = nullptr,
        .colorAttachmentCount    = 0,
        .pColorAttachments       = nullptr,
        .pResolveAttachments     = nullptr,
        .pDepthStencilAttachment = nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments    = nullptr,
    };
    return desc;
}

VkSubpassDependency subpassDependency()
{
    VkSubpassDependency dep{
        .srcSubpass      = 0,
        .dstSubpass      = 0,
        .srcStageMask    = 0, // TODO: Change to VK_PIPELINE_STAGE_NONE in 1.3
        .dstStageMask    = 0,
        .srcAccessMask   = 0, // TODO: Change to VK_ACCESS_NONE in 1.3
        .dstAccessMask   = 0,
        .dependencyFlags = 0,
    };
    return dep;
}

VkRenderPassCreateInfo renderPassCreateInfo(std::span<VkAttachmentDescription> attachments,
    VkSubpassDescription *subpass, VkSubpassDependency *dependency)
{
    VkRenderPassCreateInfo info{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
        .attachmentCount = toInt32(attachments.size()),
        .pAttachments    = attachments.data(),
        .subpassCount    = 1,
        .pSubpasses      = subpass,
        .dependencyCount = 1,
        .pDependencies   = dependency,
    };
    return info;
}

VkPipelineViewportStateCreateInfo viewportCreateInfo(VkViewport *viewport, VkRect2D *scissor)
{
    VkPipelineViewportStateCreateInfo info{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = 0, // Unused
        .viewportCount = 1,
        .pViewports    = viewport,
        .scissorCount  = 1,
        .pScissors     = scissor,
    };
    return info;
}

VkPipelineColorBlendStateCreateInfo colorBlendInfo(
    VkPipelineColorBlendAttachmentState *attachment)
{
    VkPipelineColorBlendStateCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM
        .logicOpEnable   = VK_FALSE,
        .logicOp         = VK_LOGIC_OP_NO_OP,
        .attachmentCount = 1,
        .pAttachments    = attachment,
        .blendConstants  = { 0.f, 0.f, 0.f, 0.f},
    };
    return info;
}

VkGraphicsPipelineCreateInfo pipelineCreateInfo(
    VkPipelineLayout layout, VkRenderPass render_pass)
{
    VkGraphicsPipelineCreateInfo info{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = nullptr,
        .flags               = 0, // Specify how the pipeline is created
        .stageCount          = 0,
        .pStages             = nullptr,
        .pVertexInputState   = nullptr,
        .pInputAssemblyState = nullptr,
        .pTessellationState  = nullptr,
        .pViewportState      = nullptr,
        .pRasterizationState = nullptr,
        .pMultisampleState   = nullptr,
        .pDepthStencilState  = nullptr,
        .pColorBlendState    = nullptr,
        .pDynamicState       = nullptr,
        .layout              = layout,
        .renderPass          = render_pass,
        .subpass             = 0,
        .basePipelineHandle  = VK_NULL_HANDLE,
        .basePipelineIndex   = -1,
    };
    return info;
}

} // namespace vkinit
