#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <spdlog/spdlog.h>

#include <cstdio> // stderr
#include <source_location> // std::source_location
#include <vector> // std::vector

namespace mimir::validation
{

using srcloc = std::source_location;

// Check if validation layers should be enabled
#ifdef NDEBUG
    constexpr bool enable_layers = false;
#else
    constexpr bool enable_layers = true;
#endif

// Validation layers to enable
const std::vector<const char*> layers = {
    "VK_LAYER_KHRONOS_validation"
};

constexpr void checkCuda(cudaError_t code, srcloc src = srcloc::current())
{
    if (code != cudaSuccess)
    {
        spdlog::error("CUDA assertion: {} in function {} at {}({})",
            cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
        );
    }
}

std::string getVulkanErrorString(VkResult code);

constexpr VkResult checkVulkan(VkResult code, srcloc src = srcloc::current())
{
    if (code != VK_SUCCESS)
    {
        spdlog::error("Vulkan assertion: {} in function {} at {}({})",
            getVulkanErrorString(code), src.function_name(), src.file_name(), src.line()
        );
    }
    return code;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT *p_create_info,
    const VkAllocationCallbacks *p_allocator,
    VkDebugUtilsMessengerEXT *p_debug_messenger
);

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT debug_messenger,
    const VkAllocationCallbacks *p_allocator
);

VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo();

bool checkValidationLayerSupport();

} // namespace mimir::validation