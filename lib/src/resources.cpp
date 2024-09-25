#include <mimir/engine/resources.hpp>

#include "internal/validation.hpp"

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

VkQueryPool createQueryPool(VkDevice device, uint32_t query_count)
{
    // Number of queries is twice the number of command buffers, to store space
    // for queries before and after rendering
    VkQueryPoolCreateInfo info{
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext      = nullptr,
        .flags      = 0,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = query_count, //command_buffers.size() * 2;
        .pipelineStatistics = 0,
    };

    VkQueryPool pool = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateQueryPool(device, &info, nullptr, &pool));
    vkResetQueryPool(device, pool, 0, query_count);
    return pool;
}

} // namespace mimir