#pragma once

#include <vulkan/vulkan.h>

#include <functional> // std::function
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/engine/device.hpp>

namespace mimir
{

struct VulkanQueue
{
    uint32_t family_index = ~0u;
    VkQueue queue         = VK_NULL_HANDLE;
};

struct VulkanDevice
{
    // GPU used for Vulkan operations
    PhysicalDevice physical_device{};
    VkDevice logical_device    = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VulkanQueue graphics, present;

    void initLogicalDevice(VkSurfaceKHR surface);

    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

    void generateMipmaps(VkImage image, VkFormat img_format,
        int img_width, int img_height, int mip_levels
    );
    void transitionImageLayout(VkImage image,
        VkImageLayout old_layout, VkImageLayout new_layout
    );

    std::string readMemoryHeapFlags(VkMemoryHeapFlags flags);
};

uint32_t getAlignedSize(size_t original_size, size_t min_alignment);

} // namespace mimir