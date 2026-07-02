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

    static PhysicalDevice make(VkPhysicalDevice gpu);
    DeviceMemoryStats getMemoryStats();
    VkDeviceSize getUboOffsetAlignment()
    {
        return general.properties.limits.minUniformBufferOffsetAlignment;
    };
};

// Selects a CUDA-Vulkan interop device. Pass VK_NULL_HANDLE as surface for headless
// selection (no presentation/swapchain requirement).
PhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
VkDevice createLogicalDevice(VkPhysicalDevice gpu, std::span<uint32_t> queue_families,
    bool headless = false
);

bool findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface,
    uint32_t& graphics_family, uint32_t& present_family
);

// Finds a graphics-capable queue family without requiring presentation support (headless).
bool findGraphicsQueueFamily(VkPhysicalDevice dev, uint32_t& graphics_family);

// Handle additional extensions required by CUDA interop.
// The swapchain extension is only needed for on-screen presentation; omit it for headless.
std::vector<const char*> getRequiredDeviceExtensions(bool include_swapchain = true);

// Extensions required by the path-tracing render path (LightModel::PathTracing).
std::vector<const char*> getRayTracingExtensions();

// True when the device exposes the ray tracing extensions AND features
// (acceleration structures, RT pipeline, buffer device address).
bool supportsRayTracing(VkPhysicalDevice dev);

static_assert(std::is_default_constructible_v<PhysicalDevice>);
static_assert(std::is_nothrow_default_constructible_v<PhysicalDevice>);
static_assert(std::is_trivially_default_constructible_v<PhysicalDevice>);

} // namespace mimir