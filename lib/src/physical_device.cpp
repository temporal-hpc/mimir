#include <mimir/engine/physical_device.hpp>

#include "internal/validation.hpp"

namespace mimir
{

uint32_t PhysicalDevice::findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    auto props = memory.memoryProperties;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    {
        auto flags = props.memoryTypes[i].propertyFlags;
        if ((type_filter & (1 << i)) && (flags & properties) == properties)
        {
            return i;
        }
    }
    return ~0;
}

VkFormatProperties PhysicalDevice::getFormatProperties(VkFormat format)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(handle, format, &properties);
    return properties;
}

DeviceMemoryStats PhysicalDevice::getMemoryStats()
{
    DeviceMemoryStats stats{};
    if (handle == nullptr) { return stats; }

    vkGetPhysicalDeviceMemoryProperties2(handle, &memory);
    auto props = memory.memoryProperties;
    stats.heap_count = props.memoryHeapCount;

    for (uint32_t i = 0; i < stats.heap_count; ++i)
    {
        auto heap_flags = props.memoryHeaps[i].flags;
        if (heap_flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            stats.usage  += budget.heapUsage[i];
            stats.budget += budget.heapBudget[i];
        }
    }
    return stats;
}

std::vector<PhysicalDevice> getDevices(VkInstance instance)
{
    // Get how many devices are available
    uint32_t device_count = 0;
    validation::checkVulkan(vkEnumeratePhysicalDevices(instance, &device_count, nullptr));

    // Get the handle for the vkGetPhysicalDeviceProperties2 function
    // This is needed for finding an interop-capable device
    auto fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"
    );

    // Return early if no devices were found
    std::vector<PhysicalDevice> devices;
    if (device_count < 1 || fpGetPhysicalDeviceProperties2 == nullptr) { return devices; }
    devices.reserve(device_count);

    std::vector<VkPhysicalDevice> vk_devices(device_count);
    validation::checkVulkan(vkEnumeratePhysicalDevices(instance, &device_count, vk_devices.data()));

    for (auto vk_dev : vk_devices)
    {
        PhysicalDevice device;
        device.handle = vk_dev;
        fpGetPhysicalDeviceProperties2(vk_dev, &device.general);
        vkGetPhysicalDeviceMemoryProperties2(vk_dev, &device.memory);
        vkGetPhysicalDeviceFeatures(vk_dev, &device.features);

        devices.push_back(device);
    }

    return devices;
}

} // namespace mimir