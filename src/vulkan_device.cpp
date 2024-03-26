#include "mimir/engine/vulkan_device.hpp"

#include <cstring> // memcpy
#include <set> // std::set

#include <mimir/validation.hpp>
#include "internal/vk_properties.hpp"
#include "internal/vk_initializers.hpp"

namespace mimir
{

uint32_t getAlignedSize(size_t original_size, size_t min_alignment)
{
	// Calculate required alignment based on minimum device offset alignment
	size_t aligned_size = original_size;
	if (min_alignment > 0)
    {
		aligned_size = (aligned_size + min_alignment - 1) & ~(min_alignment - 1);
	}
	return aligned_size;
}

VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool, uint32_t count)
{
    return VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = count,
    };
}

VulkanDevice::VulkanDevice(VkPhysicalDevice gpu): physical_device{gpu}
{
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    vkGetPhysicalDeviceFeatures(physical_device, &features);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    budget_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    budget_properties.pNext = nullptr;
    memory_properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memory_properties2.pNext = &budget_properties;
}

VulkanDevice::~VulkanDevice()
{
    //printf("Liberating resources...\n");
    deletors.flush();
}

void VulkanDevice::initLogicalDevice(VkSurfaceKHR surface)
{
    props::findQueueFamilies(
        physical_device, surface, graphics.family_index, present.family_index
    );

    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set unique_queue_families{ graphics.family_index, present.family_index };
    auto queue_priority = 1.f;

    for (auto queue_family : unique_queue_families)
    {
        VkDeviceQueueCreateInfo queue_create_info{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext            = nullptr,
            .flags            = 0,
            .queueFamilyIndex = queue_family,
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority,
        };
        queue_create_infos.push_back(queue_create_info);
    }

    // TODO: These features are not always used in every case,
    // so they should not be always enforced (for greater compatibility)
    VkPhysicalDeviceFeatures device_features{};
    device_features.samplerAnisotropy = VK_TRUE;
    device_features.fillModeNonSolid  = VK_TRUE; // Enable wireframe
    device_features.geometryShader    = VK_TRUE;
    device_features.shaderFloat64     = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vk12features{};
    vk12features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12features.pNext = nullptr;
    vk12features.timelineSemaphore = VK_TRUE; // Enable timeline semaphores
    vk12features.hostQueryReset    = VK_TRUE; // Enable resetting queries from host code

    VkPhysicalDeviceVulkan11Features vk11features{};
    vk11features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vk11features.pNext = &vk12features;
    vk11features.storageInputOutput16 = VK_FALSE;

    auto device_extensions = props::getRequiredDeviceExtensions();
    VkDeviceCreateInfo create_info{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &vk11features,
        .flags                   = 0,
        .queueCreateInfoCount    = vkinit::toInt32(queue_create_infos.size()),
        .pQueueCreateInfos       = queue_create_infos.data(),
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = vkinit::toInt32(device_extensions.size()),
        .ppEnabledExtensionNames = device_extensions.data(),
        .pEnabledFeatures        = &device_features,
    };
    if (validation::enable_layers)
    {
        create_info.enabledLayerCount   = validation::layers.size();
        create_info.ppEnabledLayerNames = validation::layers.data();
    }

    validation::checkVulkan(vkCreateDevice(
        physical_device, &create_info, nullptr, &logical_device)
    );
    deletors.add([=,this](){
        vkDestroyDevice(logical_device, nullptr);
    });

    vkGetDeviceQueue(logical_device, graphics.family_index, 0, &graphics.queue);
    vkGetDeviceQueue(logical_device, present.family_index, 0, &present.queue);

    command_pool = createCommandPool(graphics.family_index,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    );
}

VkCommandPool VulkanDevice::createCommandPool(
    uint32_t queue_idx, VkCommandPoolCreateFlags flags)
{
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pool_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = flags,
        .queueFamilyIndex = queue_idx,
    };
    validation::checkVulkan(vkCreateCommandPool(
        logical_device, &pool_info, nullptr, &cmd_pool)
    );
    deletors.add([=,this](){
        vkDestroyCommandPool(logical_device, command_pool, nullptr);
    });

    return cmd_pool;
}

std::vector<VkDescriptorSet> VulkanDevice::createDescriptorSets(
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
        vkAllocateDescriptorSets(logical_device, &alloc_info, sets.data())
    );
    return sets;
}

std::vector<VkCommandBuffer> VulkanDevice::createCommandBuffers(uint32_t buffer_count)
{
    std::vector<VkCommandBuffer> buffers(buffer_count, VK_NULL_HANDLE);
    auto alloc_info = commandBufferAllocateInfo(command_pool, buffer_count);
    validation::checkVulkan(vkAllocateCommandBuffers(
        logical_device, &alloc_info, buffers.data())
    );
    return buffers;
}

void VulkanDevice::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    auto queue = graphics.queue;
    VkCommandBuffer cmd;
    auto alloc_info = commandBufferAllocateInfo(command_pool, 1);
    validation::checkVulkan(vkAllocateCommandBuffers(logical_device, &alloc_info, &cmd));

    // Begin command buffer recording with a only-one-use buffer
    auto flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    auto begin_info = vkinit::commandBufferBeginInfo(flags);

    validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));
    function(cmd);
    validation::checkVulkan(vkEndCommandBuffer(cmd));

    auto submit_info = vkinit::submitInfo(&cmd);
    validation::checkVulkan(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(logical_device, command_pool, 1, &cmd);
}

uint32_t VulkanDevice::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
    {
        auto flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((type_filter & (1 << i)) && (flags & properties) == properties)
        {
            return i;
        }
    }
    return ~0;
}

VkDeviceMemory VulkanDevice::allocateMemory(VkMemoryRequirements requirements,
    VkMemoryPropertyFlags properties, const void *export_info)
{
    auto type = findMemoryType(requirements.memoryTypeBits, properties);
    VkMemoryAllocateInfo alloc_info{
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = export_info,
        .allocationSize  = requirements.size,
        .memoryTypeIndex = type,
    };

    VkDeviceMemory memory = VK_NULL_HANDLE;
    validation::checkVulkan(vkAllocateMemory(logical_device, &alloc_info, nullptr, &memory));
    return memory;
}

VkBuffer VulkanDevice::createBuffer(VkDeviceSize size,
    VkBufferUsageFlags usage, const void *extmem_info)
{
    VkBuffer buffer = VK_NULL_HANDLE;
    if (size > 0)
    {
        VkBufferCreateInfo info{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext       = extmem_info,
            .flags       = 0,
            .size        = size,
            .usage       = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices   = nullptr,
        };

        validation::checkVulkan(vkCreateBuffer(logical_device, &info, nullptr, &buffer));
    }
    return buffer;
}

VkSampler VulkanDevice::createSampler(VkFilter filter, bool enable_anisotropy)
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
        .maxAnisotropy    = properties.limits.maxSamplerAnisotropy,
        .compareEnable    = VK_FALSE,
        .compareOp        = VK_COMPARE_OP_NEVER,
        .minLod           = 0.f,
        .maxLod           = VK_LOD_CLAMP_NONE,
        .borderColor      = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VkSampler sampler = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateSampler(logical_device, &info, nullptr, &sampler));
    deletors.add([=,this]{
        vkDestroySampler(logical_device, sampler, nullptr);
    });
    return sampler;
}

VkFormat VulkanDevice::findSupportedImageFormat(const std::vector<VkFormat>& candidates,
    VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (auto format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &props);

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

void VulkanDevice::generateMipmaps(VkImage image, VkFormat img_format,
    int img_width, int img_height, int mip_levels)
{
    VkFormatProperties format_props;
    vkGetPhysicalDeviceFormatProperties(physical_device, img_format, &format_props);
    auto blit_support = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if (!(format_props.optimalTilingFeatures & blit_support))
    {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    immediateSubmit([=](VkCommandBuffer cmd)
    {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask       = 0,
            .dstAccessMask       = 0,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = image,
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            }
        };

        int32_t mip_width  = img_width;
        int32_t mip_height = img_height;

        for (uint32_t i = 1; i < static_cast<uint32_t>(mip_levels); i++)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                nullptr, 1, &barrier);

            int32_t mip_x = mip_width > 1 ? mip_width / 2 : 1;
            int32_t mip_y = mip_height > 1 ? mip_height / 2 : 1;
            VkImageBlit blit{
                .srcSubresource = VkImageSubresourceLayers{
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .srcOffsets = { {0, 0, 0}, {mip_width, mip_height, 1} },
                .dstSubresource = VkImageSubresourceLayers{
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = i,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .dstOffsets = { {0, 0, 0}, {mip_x, mip_y, 1} },
            };

            vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                            VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                                0, nullptr, 1, &barrier);

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
            0, nullptr, 1, &barrier
        );
    });
}

VkDescriptorPool VulkanDevice::createDescriptorPool(
    const std::vector<VkDescriptorPoolSize>& pool_sizes)
{
    VkDescriptorPoolCreateInfo pool_info{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets       = 1000,
        .poolSizeCount = vkinit::toInt32(pool_sizes.size()),
        .pPoolSizes    = pool_sizes.data(),
    };

    VkDescriptorPool pool = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateDescriptorPool(logical_device, &pool_info, nullptr, &pool)
    );
    deletors.add([=,this]{
        vkDestroyDescriptorPool(logical_device, pool, nullptr);
    });
    return pool;
}

VkDescriptorSetLayout VulkanDevice::createDescriptorSetLayout(
    const std::vector<VkDescriptorSetLayoutBinding>& layout_bindings)
{
    VkDescriptorSetLayoutCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext        = nullptr,
        .flags        = 0,
        .bindingCount = vkinit::toInt32(layout_bindings.size()),
        .pBindings    = layout_bindings.data(),
    };

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateDescriptorSetLayout(logical_device, &info, nullptr, &layout)
    );
    deletors.add([=,this](){
        vkDestroyDescriptorSetLayout(logical_device, layout, nullptr);
    });
    return layout;
}

VkPipelineLayout VulkanDevice::createPipelineLayout(VkDescriptorSetLayout descriptor_layout)
{
    std::vector<VkDescriptorSetLayout> layouts{descriptor_layout};
    VkPipelineLayoutCreateInfo info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext                  = nullptr,
        .flags                  = 0, // Currently unused
        .setLayoutCount         = vkinit::toInt32(layouts.size()),
        .pSetLayouts            = layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges    = nullptr,
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreatePipelineLayout(logical_device, &info, nullptr, &layout));
    deletors.add([=,this]{ vkDestroyPipelineLayout(logical_device, layout, nullptr); });
    return layout;
}

VkFence VulkanDevice::createFence(VkFenceCreateFlags flags)
{
    VkFenceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
    };
    VkFence fence = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateFence(logical_device, &info, nullptr, &fence));
    deletors.add([=,this]{
        vkDestroyFence(logical_device, fence, nullptr);
    });
    return fence;
}

VkSemaphore VulkanDevice::createSemaphore(const void *extensions)
{
    auto info = vkinit::semaphoreCreateInfo(extensions);
    VkSemaphore semaphore = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateSemaphore(logical_device, &info, nullptr, &semaphore));
    deletors.add([=,this]{
        vkDestroySemaphore(logical_device, semaphore, nullptr);
    });
    return semaphore;
}

void VulkanDevice::transitionImageLayout(VkImage image,
    VkImageLayout old_layout, VkImageLayout new_layout)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;

    VkPipelineStageFlags src_stage, dst_stage;
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        barrier.srcAccessMask = 0;
        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition");
    }

    if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition");
    }

    immediateSubmit([=](VkCommandBuffer cmd)
    {
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    });
}

ConvertedMemory VulkanDevice::formatMemory(uint64_t memsize) const
{
    constexpr float kilobyte = 1024.f;
    constexpr float megabyte = kilobyte * 1024.f;
    constexpr float gigabyte = megabyte * 1024.f;

    ConvertedMemory converted{};
    converted.data = static_cast<float>(memsize) / gigabyte;
    converted.units = "GB";

    return converted;
}

std::string VulkanDevice::readMemoryHeapFlags(VkMemoryHeapFlags flags)
{
    switch (flags)
    {
        case VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: return "Device local bit";
        case VK_MEMORY_HEAP_MULTI_INSTANCE_BIT: return "Multiple instance bit";
        default: return "Host local heap memory";
    }
    return "";
}

void VulkanDevice::updateMemoryProperties()
{
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory_properties2);
    props.heap_count = memory_properties2.memoryProperties.memoryHeapCount;
    props.gpu_usage = 0;
    props.gpu_budget = 0;

    for (uint32_t i = 0; i < props.heap_count; ++i)
    {
        auto heap_flags = memory_properties2.memoryProperties.memoryHeaps[i].flags;
        if (heap_flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            props.gpu_usage += budget_properties.heapUsage[i];
            props.gpu_budget += budget_properties.heapBudget[i];
        }
    }
}

void VulkanDevice::listExtensions()
{
    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, available.data());

    printf("Available extensions:\n");
    for (const auto& extension : available)
    {
        printf("  %s\n", extension.extensionName);
    }
}

VkQueryPool VulkanDevice::createQueryPool(uint32_t query_count)
{
    VkQueryPoolCreateInfo info{
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext      = nullptr,
        .flags      = 0,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        // Number of queries is twice the number of command buffers, to store space
        // for queries before and after rendering
        .queryCount = query_count, //command_buffers.size() * 2;
        .pipelineStatistics = 0,
    };

    VkQueryPool pool = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateQueryPool(logical_device, &info, nullptr, &pool));
    deletors.add([=,this]{
        vkDestroyQueryPool(logical_device, pool, nullptr);
    });
    vkResetQueryPool(logical_device, pool, 0, query_count);
    return pool;
}

} // namespace mimir