#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

namespace mimir
{

// Aggregate structure containing a Vulkan physical device handle and its
// associated properties
// Default values are set for ready use by the various Vulkan device property
// retrieval functions, filling the empty values below with device information
struct PhysicalDevice
{
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VkPhysicalDeviceIDProperties id_props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
        .pNext = nullptr,
        .deviceUUID = {},
        .driverUUID = {},
        .deviceLUID = {},
        .deviceNodeMask  = {},
        .deviceLUIDValid = {},
    };
    VkPhysicalDeviceProperties2 general{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &id_props,
        .properties = {},
    };
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
        .pNext = nullptr,
        .heapBudget = {},
        .heapUsage  = {},
    };
    VkPhysicalDeviceMemoryProperties2 memory{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = &budget,
        .memoryProperties = {},
    };
    VkPhysicalDeviceFeatures features{};
};

std::vector<PhysicalDevice> getDevices(VkInstance instance);

} // namespace mimir