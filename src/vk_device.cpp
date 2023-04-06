#include "cudaview/engine/vk_device.hpp"

#include <set> // std::set

#include "internal/utils.hpp"
#include <cudaview/validation.hpp>
#include "internal/vk_properties.hpp"
#include "internal/vk_initializers.hpp"

VulkanDevice::VulkanDevice(VkPhysicalDevice gpu): physical_device{gpu}
{
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    vkGetPhysicalDeviceFeatures(physical_device, &features);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
}

VulkanDevice::~VulkanDevice()
{
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
        VkDeviceQueueCreateInfo queue_create_info{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family;
        queue_create_info.queueCount       = 1;
        queue_create_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_create_info);
    }

    VkPhysicalDeviceFeatures device_features{};
    device_features.samplerAnisotropy = VK_TRUE;
    device_features.fillModeNonSolid  = VK_TRUE; // Enable wireframe
    device_features.geometryShader    = VK_TRUE;

    // Explicitly enable timeline semaphores, or validation layer will complain
    VkPhysicalDeviceVulkan12Features features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features.timelineSemaphore = true;

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &features;
    create_info.queueCreateInfoCount = queue_create_infos.size();
    create_info.pQueueCreateInfos    = queue_create_infos.data();
    create_info.pEnabledFeatures     = &device_features;

    auto device_extensions = props::getRequiredDeviceExtensions();
    create_info.enabledExtensionCount   = device_extensions.size();
    create_info.ppEnabledExtensionNames = device_extensions.data();

    if (validation::enable_layers)
    {
        create_info.enabledLayerCount   = validation::layers.size();
        create_info.ppEnabledLayerNames = validation::layers.data();
    }
    else
    {
        create_info.enabledLayerCount = 0;
    }

    validation::checkVulkan(vkCreateDevice(
        physical_device, &create_info, nullptr, &logical_device)
    );
    deletors.pushFunction([=](){
        printf("Destroying device\n");
        vkDestroyDevice(logical_device, nullptr);
    });

    vkGetDeviceQueue(logical_device, graphics.family_index, 0, &graphics.queue);
    vkGetDeviceQueue(logical_device, present.family_index, 0, &present.queue);

    // TODO: Get device UUID for cuda

    command_pool = createCommandPool(graphics.family_index,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    );
}

VkCommandPool VulkanDevice::createCommandPool(
    uint32_t queue_idx, VkCommandPoolCreateFlags flags)
{
    VkCommandPool new_pool = VK_NULL_HANDLE;
    auto pool_info = vkinit::commandPoolCreateInfo(flags, queue_idx);
    validation::checkVulkan(vkCreateCommandPool(
        logical_device, &pool_info, nullptr, &new_pool)
    );
    deletors.pushFunction([=](){
        vkDestroyCommandPool(logical_device, command_pool, nullptr);
    });

    return new_pool;
}

std::vector<VkDescriptorSet> VulkanDevice::createDescriptorSets(
    VkDescriptorPool pool, VkDescriptorSetLayout layout, uint32_t set_count)
{
    std::vector<VkDescriptorSetLayout> layouts(set_count, layout);
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool     = pool;
    alloc_info.descriptorSetCount = set_count;
    alloc_info.pSetLayouts        = layouts.data();

    std::vector<VkDescriptorSet> sets(set_count, VK_NULL_HANDLE);
    validation::checkVulkan(
        vkAllocateDescriptorSets(logical_device, &alloc_info, sets.data())
    );
    return sets;
}

std::vector<VkCommandBuffer> VulkanDevice::createCommandBuffers(uint32_t buffer_count)
{
    std::vector<VkCommandBuffer> buffers(buffer_count, VK_NULL_HANDLE);
    auto alloc_info = vkinit::commandBufferAllocateInfo(
        command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, buffer_count
    );
    validation::checkVulkan(vkAllocateCommandBuffers(
        logical_device, &alloc_info, buffers.data())
    );
    return buffers;
}

void VulkanDevice::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    auto queue = graphics.queue;
    VkCommandBuffer cmd;
    auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool);
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
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = export_info;
    alloc_info.allocationSize = requirements.size;
    auto type = findMemoryType(requirements.memoryTypeBits, properties);
    alloc_info.memoryTypeIndex = type;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    validation::checkVulkan(vkAllocateMemory(logical_device, &alloc_info, nullptr, &memory));
    return memory;
}

VkBuffer VulkanDevice::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
    const void *extmem_info)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size  = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.pNext = extmem_info;

    VkBuffer buffer = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateBuffer(logical_device, &info, nullptr, &buffer));
    return buffer;
}

VkImage VulkanDevice::createImage(VkImageType type, VkFormat format,
    VkExtent3D extent, VkImageTiling tiling, VkImageUsageFlags usage,
    const void *extmem_info)
{
    // TODO: Check if texture is within bounds
    //auto max_img_dim = properties.limits.maxImageDimension3D;

    auto info = vkinit::imageCreateInfo(type, format, extent, usage);
    info.pNext         = extmem_info;
    info.flags         = 0;
    info.tiling        = tiling;
    info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(logical_device, &info, nullptr, &image));
    return image;
}

VkSampler VulkanDevice::createSampler(VkFilter filter, bool enable_anisotropy)
{
    auto info = vkinit::samplerCreateInfo(filter);
    info.anisotropyEnable = enable_anisotropy? VK_TRUE : VK_FALSE;
    info.maxAnisotropy    = properties.limits.maxSamplerAnisotropy;

    VkSampler sampler;
    validation::checkVulkan(vkCreateSampler(logical_device, &info, nullptr, &sampler));
    deletors.pushFunction([=]{
        vkDestroySampler(logical_device, sampler, nullptr);
    });
    return sampler;
}

void VulkanDevice::generateMipmaps(VkImage image, VkFormat img_format,
    int img_width, int img_height, int mip_levels)
{
    VkFormatProperties format_props;
    vkGetPhysicalDeviceFormatProperties(physical_device, img_format, &format_props);

    if (!(format_props.optimalTilingFeatures &
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
    {
        throw std::runtime_error(
        "texture image format does not support linear blitting!");
    }

    immediateSubmit([=](VkCommandBuffer cmd)
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mip_width  = img_width;
        int32_t mip_height = img_height;

        for (int i = 1; i < mip_levels; i++)
        {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                            nullptr, 1, &barrier);

        VkImageBlit blit = {};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mip_width, mip_height, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mip_width > 1 ? mip_width / 2 : 1,
                                mip_height > 1 ? mip_height / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

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
        0, nullptr, 1, &barrier);
    });
}

VkDescriptorPool VulkanDevice::createDescriptorPool(
    const std::vector<VkDescriptorPoolSize>& pool_sizes)
{
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets       = 1000;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes    = pool_sizes.data();

    VkDescriptorPool pool;
    validation::checkVulkan(
        vkCreateDescriptorPool(logical_device, &pool_info, nullptr, &pool)
    );
    deletors.pushFunction([=]{
        vkDestroyDescriptorPool(logical_device, pool, nullptr);
    });
    return pool;
}

VkMemoryRequirements VulkanDevice::getMemoryRequiements(VkBufferUsageFlags usage,
    const std::vector<uint32_t>& sizes)
{
    VkMemoryRequirements reqs;

    // Test buffer for asking about its memory properties
    auto test_buffer = createBuffer(1, usage);
    vkGetBufferMemoryRequirements(logical_device, test_buffer, &reqs);
    // Get maximum aligned size to set it in the requirements struct
    uint32_t max_aligned_size = 0;
    for (auto size : sizes)
    {
        auto aligned_size = getAlignedSize(size, reqs.alignment);
        if (aligned_size > max_aligned_size) max_aligned_size = aligned_size;
    }
    reqs.size = max_aligned_size;
    // Destroy the test buffer, since a proper one should be created with
    // the returned requirements 
    vkDestroyBuffer(logical_device, test_buffer, nullptr);
    
    return reqs;
}

VkDescriptorSetLayout VulkanDevice::createDescriptorSetLayout(
    const std::vector<VkDescriptorSetLayoutBinding>& layout_bindings)
{
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = layout_bindings.size();
    info.pBindings    = layout_bindings.data();

    VkDescriptorSetLayout layout;
    validation::checkVulkan(
        vkCreateDescriptorSetLayout(logical_device, &info, nullptr, &layout)
    );
        deletors.pushFunction([=](){
            vkDestroyDescriptorSetLayout(logical_device, layout, nullptr);
        });
    return layout;
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
