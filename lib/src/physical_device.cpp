#include "internal/physical_device.hpp"

#include <mimir/validation.hpp>

namespace mimir
{

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