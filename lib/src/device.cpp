#include "mimir/device.hpp"

#include <spdlog/spdlog.h>
#include <set> // std::set

#include "mimir/validation.hpp"

namespace mimir
{

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

PhysicalDevice PhysicalDevice::make(VkPhysicalDevice gpu)
{
    PhysicalDevice dev{
        .handle = gpu,
        .id_props{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
            .pNext = nullptr,
            .deviceUUID = {},
            .driverUUID = {},
            .deviceLUID = {},
            .deviceNodeMask  = {},
            .deviceLUIDValid = {},
        },
        .general{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &dev.id_props,
            .properties = {},
        },
        .budget{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
            .pNext = nullptr,
            .heapBudget = {},
            .heapUsage  = {},
        },
        .memory{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
            .pNext = &dev.budget,
            .memoryProperties = {},
        },
        .features{},
    };

    vkGetPhysicalDeviceProperties2(gpu, &dev.general);
    vkGetPhysicalDeviceMemoryProperties2(gpu, &dev.memory);
    vkGetPhysicalDeviceFeatures(gpu, &dev.features);
    return dev;
}

DeviceMemoryStats PhysicalDevice::getMemoryStats()
{
    DeviceMemoryStats stats{ .heap_count = 0, .usage  = 0, .budget = 0 };
    if (handle == nullptr) { return stats; }

    memory.pNext = &budget;
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

    // Return early if no devices were found
    std::vector<PhysicalDevice> devices;
    if (device_count < 1) { return devices; }
    devices.reserve(device_count);

    // Create physical device structures
    std::vector<VkPhysicalDevice> vk_devs(device_count);
    validation::checkVulkan(vkEnumeratePhysicalDevices(instance, &device_count, vk_devs.data()));
    for (auto vk_dev : vk_devs) { devices.push_back(PhysicalDevice::make(vk_dev)); }
    return devices;
}

// Logic to find queue family indices to populate struct with
bool findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface,
    uint32_t& graphics_family, uint32_t& present_family)
{
    constexpr auto family_empty = ~0u;
    // Assign index to queue families that could be found
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &family_count, families.data());

    graphics_family = present_family = family_empty;

    // Find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
    for (uint32_t i = 0; i < family_count; ++i)
    {
        auto family = families[i];
        if (family.queueCount > 0)
        {
            if (graphics_family == family_empty && family.queueFlags & VK_QUEUE_GRAPHICS_BIT
                && family.timestampValidBits > 0)
            {
                graphics_family = i;
            }
            uint32_t present_support = 0;
            validation::checkVulkan(vkGetPhysicalDeviceSurfaceSupportKHR(
                dev, i, surface, &present_support)
            );
            if (present_family == family_empty && present_support)
            {
                present_family = i;
            }
            if (present_family != family_empty && graphics_family != family_empty)
            {
                break;
            }
        }
    }
    return graphics_family != family_empty && present_family != family_empty;
}

std::vector<const char*> getRequiredDeviceExtensions()
{
    return std::vector<const char*>{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    };
}

bool checkAllExtensionsSupported(VkPhysicalDevice dev, std::span<const char*> expected)
{
    // Enumerate extensions and check if all required extensions are included
    uint32_t ext_count;
    validation::checkVulkan(vkEnumerateDeviceExtensionProperties(
        dev, nullptr, &ext_count, nullptr)
    );
    std::vector<VkExtensionProperties> available(ext_count);
    validation::checkVulkan(vkEnumerateDeviceExtensionProperties(
        dev, nullptr, &ext_count, available.data())
    );

    std::set<std::string> required( expected.begin(), expected.end() );
    for (const auto& extension : available) {required.erase(extension.extensionName); }
    return required.empty();
}

SwapchainSupportDetails getSwapchainProperties(VkPhysicalDevice dev, VkSurfaceKHR surface)
{
    SwapchainSupportDetails details;
    validation::checkVulkan(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        dev, surface, &details.capabilities
    ));

    uint32_t format_count;
    validation::checkVulkan(vkGetPhysicalDeviceSurfaceFormatsKHR(
        dev, surface, &format_count, nullptr
    ));
    if (format_count > 0)
    {
        details.formats.resize(format_count);
        validation::checkVulkan(vkGetPhysicalDeviceSurfaceFormatsKHR(
            dev, surface, &format_count, details.formats.data()
        ));
    }

    uint32_t mode_count;
    validation::checkVulkan(vkGetPhysicalDeviceSurfacePresentModesKHR(
        dev, surface, &mode_count, nullptr
    ));
    if (mode_count != 0)
    {
        details.present_modes.resize(mode_count);
        validation::checkVulkan(vkGetPhysicalDeviceSurfacePresentModesKHR(
            dev, surface, &mode_count, details.present_modes.data()
        ));
    }
    return details;
}

bool isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface)
{
    uint32_t graphics_idx, present_idx;
    auto has_queues          = findQueueFamilies(dev, surface, graphics_idx, present_idx);
    auto device_extensions   = getRequiredDeviceExtensions();
    auto supports_extensions = checkAllExtensionsSupported(dev, device_extensions);
    auto swapchain_support   = getSwapchainProperties(dev, surface);
    auto swapchain_adequate  = !swapchain_support.formats.empty() &&
                               !swapchain_support.present_modes.empty();
    VkPhysicalDeviceFeatures supported_features;
    vkGetPhysicalDeviceFeatures(dev, &supported_features);
    return supports_extensions && swapchain_adequate && has_queues
        && supported_features.samplerAnisotropy;
}

PhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface)
{
    int cuda_dev_count = 0;
    validation::checkCuda(cudaGetDeviceCount(&cuda_dev_count));
    if (cuda_dev_count == 0)
    {
        spdlog::error("could not find devices supporting CUDA");
    }
    auto all_devices = getDevices(instance);

#ifndef NDEBUG
    printf("Enumerating CUDA devices:\n");
    for (int dev_id = 0; dev_id < cuda_dev_count; ++dev_id)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, dev_id);
        printf("* ID: %d\n  Name: %s\n  Capability: %d.%d\n",
            dev_id, dev_prop.name, dev_prop.major, dev_prop.minor
        );
    }
    printf("Enumerating Vulkan devices:\n");
    for (const auto& dev : all_devices)
    {
        auto props = dev.general.properties;
        printf("* ID: %u\n  Name: %s\n", props.deviceID, props.deviceName);
    }
#endif

    PhysicalDevice chosen_device{};
    int curr_device = 0, prohibited_count = 0;
    while (curr_device < cuda_dev_count)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, curr_device);
        if (dev_prop.computeMode == cudaComputeModeProhibited)
        {
            prohibited_count++;
            curr_device++;
            continue;
        }
        for (const auto& device : all_devices)
        {
            auto uuid = device.id_props.deviceUUID;
            auto matching = memcmp((void*)&dev_prop.uuid, uuid, VK_UUID_SIZE) == 0;
            if (matching && isDeviceSuitable(device.handle, surface))
            {
                validation::checkCuda(cudaSetDevice(curr_device));
                auto device_name = device.general.properties.deviceName;
                spdlog::info("Selected interop device {}: {}", curr_device, device_name);
                chosen_device = device;
                break;
            }
        }
        curr_device++;
    }

    if (prohibited_count == cuda_dev_count)
    {
        spdlog::error("No CUDA-Vulkan interop device was found");
    }
    return chosen_device;
}

VkDevice createLogicalDevice(VkPhysicalDevice gpu, std::span<uint32_t> queue_families)
{
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    auto queue_priority = 1.f;
    for (auto queue_family : queue_families)
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

    VkDevice device = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateDevice(gpu, &create_info, nullptr, &device));
    return device;
}

void listExtensions()
{
    uint32_t ext_count = 0;
    validation::checkVulkan(vkEnumerateInstanceExtensionProperties(
        nullptr, &ext_count, nullptr)
    );
    std::vector<VkExtensionProperties> available(ext_count);
    validation::checkVulkan(vkEnumerateInstanceExtensionProperties(
        nullptr, &ext_count, available.data())
    );

    printf("Available extensions:\n");
    for (const auto& extension : available) { printf("  %s\n", extension.extensionName); }
}

} // namespace mimir