#include <mimir/engine/vulkan_device.hpp>

#include <set> // std::set

#include <mimir/engine/device.hpp>
#include <mimir/engine/resources.hpp>
#include "internal/validation.hpp"

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

void VulkanDevice::initLogicalDevice(VkSurfaceKHR surface)
{
    findQueueFamilies(physical_device.handle, surface, graphics.family_index, present.family_index);

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

    auto device_extensions = getRequiredDeviceExtensions();
    VkDeviceCreateInfo create_info{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &vk11features,
        .flags                   = 0,
        .queueCreateInfoCount    = (uint32_t)queue_create_infos.size(),
        .pQueueCreateInfos       = queue_create_infos.data(),
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = (uint32_t)device_extensions.size(),
        .ppEnabledExtensionNames = device_extensions.data(),
        .pEnabledFeatures        = &device_features,
    };
    if (validation::enable_layers)
    {
        create_info.enabledLayerCount   = validation::layers.size();
        create_info.ppEnabledLayerNames = validation::layers.data();
    }

    validation::checkVulkan(vkCreateDevice(
        physical_device.handle, &create_info, nullptr, &logical_device)
    );

    vkGetDeviceQueue(logical_device, graphics.family_index, 0, &graphics.queue);
    vkGetDeviceQueue(logical_device, present.family_index, 0, &present.queue);

    command_pool = createCommandPool(logical_device, graphics.family_index,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    );
}

void VulkanDevice::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VkCommandBuffer cmd;
    auto alloc_info = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    validation::checkVulkan(vkAllocateCommandBuffers(logical_device, &alloc_info, &cmd));

    // Begin command buffer recording with a only-one-use buffer
    VkCommandBufferBeginInfo cmd_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    validation::checkVulkan(vkBeginCommandBuffer(cmd, &cmd_info));
    function(cmd);
    validation::checkVulkan(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit_info{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = nullptr,
        .waitSemaphoreCount   = 0,
        .pWaitSemaphores      = nullptr,
        .pWaitDstStageMask    = nullptr,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &cmd,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores    = nullptr,
    };
    auto queue = graphics.queue;
    validation::checkVulkan(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(logical_device, command_pool, 1, &cmd);
}

void VulkanDevice::generateMipmaps(VkImage image, VkFormat format,
    int img_width, int img_height, int mip_levels)
{
    auto props = getImageFormatProperties(physical_device.handle, format);
    auto blit_support = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if (!(props.optimalTilingFeatures & blit_support))
    {
        spdlog::error("texture image format does not support linear blitting!");
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
        spdlog::error("unsupported layout transition");
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
        spdlog::error("unsupported layout transition");
    }

    immediateSubmit([=](VkCommandBuffer cmd)
    {
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    });
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

} // namespace mimir