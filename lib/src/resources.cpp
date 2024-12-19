#include "mimir/resources.hpp"

#include <spdlog/spdlog.h>
#include "mimir/validation.hpp"

namespace mimir
{

uint32_t findMemoryType(VkPhysicalDeviceMemoryProperties available,
    uint32_t type_filter, VkMemoryPropertyFlags requested)
{
    for (uint32_t i = 0; i < available.memoryTypeCount; ++i)
    {
        auto flags = available.memoryTypes[i].propertyFlags;
        if ((type_filter & (1 << i)) && (flags & requested) == requested)
        {
            return i;
        }
    }
    return ~0;
}

VkDeviceMemory allocateMemory(VkDevice device, VkPhysicalDeviceMemoryProperties properties,
    VkMemoryRequirements requirements, VkMemoryPropertyFlags flags, const void *export_info)
{
    VkMemoryAllocateInfo alloc_info{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = export_info,
        .allocationSize  = requirements.size,
        .memoryTypeIndex = findMemoryType(properties, requirements.memoryTypeBits, flags),
    };

    VkDeviceMemory memory = VK_NULL_HANDLE;
    validation::checkVulkan(vkAllocateMemory(device, &alloc_info, nullptr, &memory));
    return memory;
}

VkBuffer createBuffer(VkDevice device, VkDeviceSize size,
    VkBufferUsageFlags usage, const void *extensions)
{
    VkBufferCreateInfo info{
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = extensions,
        .flags                 = 0,
        .size                  = size,
        .usage                 = usage,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
    };
    VkBuffer buffer = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateBuffer(device, &info, nullptr, &buffer));
    return buffer;
}

VkFormatProperties getImageFormatProperties(VkPhysicalDevice ph_dev, VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(ph_dev, format, &properties);
    return properties;
}

VkFormat findSupportedImageFormat(VkPhysicalDevice ph_dev, std::span<VkFormat> candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (auto format : candidates)
    {
        auto props = getImageFormatProperties(ph_dev, format);
        switch (tiling)
        {
            case VK_IMAGE_TILING_LINEAR:
            {
                if ((props.linearTilingFeatures & features) == features) return format;
                break;
            }
            case VK_IMAGE_TILING_OPTIMAL:
            {
                if ((props.optimalTilingFeatures & features) == features) return format;
                break;
            }
            default: return VK_FORMAT_UNDEFINED;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

VkImage createImage(VkDevice device, VkPhysicalDevice ph_dev, ImageParams params, const void *extensions)
{
    // TODO: Also check if texture is within bounds
    // auto max_dim = getMaxImageDimension(params.layout);
    // if (extent.width >= max_dim || extent.height >= max_dim || extent.height >= max_dim)
    // {
    //     spdlog::error("Requested image dimensions are larger than maximum");
    // }

    // Check that a Vulkan image handle can be created with the supplied parameters
    VkPhysicalDeviceImageFormatInfo2 format_info{
        .sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext  = nullptr,
        .format = params.format,
        .type   = params.type,
        .tiling = params.tiling,
        .usage  = params.usage,
        .flags  = 0
    };
    VkImageFormatProperties2 format_props{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = nullptr,
        .imageFormatProperties = {} // To be filled in function below
    };
    validation::checkVulkan(vkGetPhysicalDeviceImageFormatProperties2(
        ph_dev, &format_info, &format_props
    ));

    VkImageCreateInfo info{
        .sType                 = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext                 = extensions,
        .flags                 = 0,
        .imageType             = params.type,
        .format                = params.format,
        .extent                = params.extent,
        .mipLevels             = params.levels,
        .arrayLayers           = 1,
        .samples               = VK_SAMPLE_COUNT_1_BIT,
        .tiling                = params.tiling,
        .usage                 = params.usage,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(device, &info, nullptr, &image));
    return image;
}

VkImageView createImageView(VkDevice device, VkImage image, ImageParams params, VkImageAspectFlags flags)
{
    VkImageViewCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = params.format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = flags,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    VkImageView view = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImageView(device, &info, nullptr, &view));
    return view;
}

VkCommandPool createCommandPool(VkDevice device,
    uint32_t queue_idx, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .queueFamilyIndex = queue_idx,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateCommandPool(device, &info, nullptr, &pool));
    return pool;
}

std::vector<VkCommandBuffer> createCommandBuffers(VkDevice device,
    VkCommandPool command_pool, uint32_t buffer_count)
{
    VkCommandBufferAllocateInfo alloc_info{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = buffer_count,
    };
    std::vector<VkCommandBuffer> buffers(buffer_count, VK_NULL_HANDLE);
    validation::checkVulkan(vkAllocateCommandBuffers(
        device, &alloc_info, buffers.data())
    );
    return buffers;
}

VkDescriptorPool createDescriptorPool(VkDevice device,
    std::span<VkDescriptorPoolSize> pool_sizes)
{
    VkDescriptorPoolCreateInfo pool_info{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets       = 1000,
        .poolSizeCount = (uint32_t)pool_sizes.size(),
        .pPoolSizes    = pool_sizes.data(),
    };

    VkDescriptorPool pool = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateDescriptorPool(device, &pool_info, nullptr, &pool)
    );
    return pool;
}

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device,
    std::span<VkDescriptorSetLayoutBinding> bindings)
{
    VkDescriptorSetLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = nullptr,
        .flags        = 0,
        .bindingCount = (uint32_t)bindings.size(),
        .pBindings    = bindings.data(),
    };

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateDescriptorSetLayout(device, &info, nullptr, &layout)
    );
    return layout;
}

std::vector<VkDescriptorSet> createDescriptorSets(VkDevice device,
    VkDescriptorPool pool, VkDescriptorSetLayout layout, uint32_t set_count)
{
    std::vector<VkDescriptorSetLayout> layouts(set_count, layout);
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext              = nullptr,
        .descriptorPool     = pool,
        .descriptorSetCount = set_count,
        .pSetLayouts        = layouts.data(),
    };

    std::vector<VkDescriptorSet> sets(set_count, VK_NULL_HANDLE);
    validation::checkVulkan(
        vkAllocateDescriptorSets(device, &alloc_info, sets.data())
    );
    return sets;
}

VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout descriptor_layout)
{
    std::vector<VkDescriptorSetLayout> layouts{descriptor_layout};
    VkPipelineLayoutCreateInfo info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0, // Currently unused
        .setLayoutCount         = (uint32_t)layouts.size(),
        .pSetLayouts            = layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges    = nullptr,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreatePipelineLayout(device, &info, nullptr, &layout));
    return layout;
}

VkSampler createSampler(VkDevice device, VkFilter filter, bool enable_anisotropy)
{
    VkSamplerCreateInfo info{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .magFilter        = filter,
        .minFilter        = filter,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias       = 0.f,
        .anisotropyEnable = enable_anisotropy? VK_TRUE : VK_FALSE,
        .maxAnisotropy    = 0.f, // TODO: physical_device.general.properties.limits.maxSamplerAnisotropy,
        .compareEnable    = VK_FALSE,
        .compareOp        = VK_COMPARE_OP_NEVER,
        .minLod           = 0.f,
        .maxLod           = VK_LOD_CLAMP_NONE,
        .borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VkSampler sampler = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateSampler(device, &info, nullptr, &sampler));
    return sampler;
}

VkFence createFence(VkDevice device, VkFenceCreateFlags flags)
{
    VkFenceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
    };
    VkFence fence = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateFence(device, &info, nullptr, &fence));
    return fence;
}


VkSemaphore createSemaphore(VkDevice device, const void *extensions)
{
    VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = extensions,
        .flags = 0, // Unused
    };
    VkSemaphore semaphore = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateSemaphore(device, &info, nullptr, &semaphore));
    return semaphore;
}

VkRenderPass createRenderPass(VkDevice device,
    VkAttachmentDescription color, VkAttachmentDescription depth)
{
    VkAttachmentReference color_ref{
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentReference depth_ref{
        .attachment = 1,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription subpass{
        .flags                   = 0, // Specify subpass usage
        .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount    = 0,
        .pInputAttachments       = nullptr,
        .colorAttachmentCount    = 1,
        .pColorAttachments       = &color_ref,
        .pResolveAttachments     = nullptr,
        .pDepthStencilAttachment = &depth_ref,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments    = nullptr,
    };

    // Specify memory and execution dependencies between subpasses
    VkPipelineStageFlags stage_mask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    VkAccessFlags access_mask =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkSubpassDependency dependency{
        .srcSubpass      = VK_SUBPASS_EXTERNAL,
        .dstSubpass      = 0,
        .srcStageMask    = stage_mask,
        .dstStageMask    = stage_mask,
        .srcAccessMask   = 0, // TODO: Change to VK_ACCESS_NONE in 1.3
        .dstAccessMask   = access_mask,
        .dependencyFlags = 0,
    };

    VkAttachmentDescription attachments[2] = { color, depth };
    VkRenderPassCreateInfo info{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
        .attachmentCount = (uint32_t)std::size(attachments),
        .pAttachments    = attachments,
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &dependency,
    };

    VkRenderPass render_pass = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateRenderPass(device, &info, nullptr, &render_pass));
    return render_pass;
}

} // namespace mimir