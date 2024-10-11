#pragma once

#include <vulkan/vulkan.h>

#include <span> // std::span
#include <vector> // std::vector

namespace mimir
{

struct DeviceMemoryStats
{
    uint32_t heap_count;
    VkDeviceSize usage;
    VkDeviceSize budget;
};

// Aggregate structure containing a Vulkan physical device handle and its associated properties
// Default values are set for ready use by the various Vulkan device property
// retrieval functions, filling the empty values below with device information
struct PhysicalDevice
{
    VkPhysicalDevice handle;
    VkPhysicalDeviceIDProperties id_props;
    VkPhysicalDeviceProperties2 general;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget;
    VkPhysicalDeviceMemoryProperties2 memory;
    VkPhysicalDeviceFeatures features;

    static PhysicalDevice make(VkPhysicalDevice device);
    DeviceMemoryStats getMemoryStats();
    VkDeviceSize getUboOffsetAlignment()
    {
        return general.properties.limits.minUniformBufferOffsetAlignment;
    };
};

PhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
VkDevice createLogicalDevice(VkPhysicalDevice gpu, std::span<uint32_t> queue_families);

bool findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface,
    uint32_t& graphics_family, uint32_t& present_family
);

// Handle additional extensions required by CUDA interop
std::vector<const char*> getRequiredDeviceExtensions();

static_assert(std::is_default_constructible_v<PhysicalDevice>);
static_assert(std::is_nothrow_default_constructible_v<PhysicalDevice>);
static_assert(std::is_trivially_default_constructible_v<PhysicalDevice>);

} // namespace mimir