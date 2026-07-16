#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <spdlog/spdlog.h>

#include <cstdio> // stderr
#include <cstdlib> // std::abort on unrecoverable Vulkan errors
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
        // Unrecoverable errors (allocation failures, lost device) would otherwise continue with a
        // null handle and segfault later; fail fast with the logged message instead. Recoverable
        // codes (e.g. VK_ERROR_OUT_OF_DATE_KHR on resize) are left for the caller to handle.
        if (code == VK_ERROR_OUT_OF_DEVICE_MEMORY || code == VK_ERROR_OUT_OF_HOST_MEMORY ||
            code == VK_ERROR_DEVICE_LOST)
        {
            // spdlog is silenced in release builds (level off), so the message above is invisible and
            // this becomes an opaque SIGABRT. Print the fatal reason straight to stderr so an OOM
            // (e.g. a grid too large for VRAM, or transient pressure from a not-yet-reclaimed process)
            // is always diagnosable.
            fprintf(stderr, "mimir fatal: %s in %s at %s(%u)\n",
                getVulkanErrorString(code).c_str(), src.function_name(), src.file_name(),
                static_cast<unsigned>(src.line()));
            std::abort();
        }
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