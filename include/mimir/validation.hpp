#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>
#define SLANG_CUDA_ENABLE_HALF
#include <slang.h>

#include <cstdio> // stderr
#include <experimental/source_location> // std::source_location
#include <stdexcept> // std::throw
#include <vector> // std::vector

namespace mimir
{
namespace validation
{

using source_location = std::experimental::source_location;

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

constexpr void checkCuda(cudaError_t code, bool panic = true,
    source_location src = source_location::current())
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA assertion: %s in function %s at %s(%d)\n",
            cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            throw std::runtime_error("CUDA failure!");
        }
    }
}

std::string getVulkanErrorString(VkResult code);

constexpr VkResult checkVulkan(VkResult code, bool panic = true,
    source_location src = source_location::current())
{
    if (code != VK_SUCCESS)
    {
        fprintf(stderr, "Vulkan assertion: %s in function %s at %s(%d)\n",
            getVulkanErrorString(code).c_str(),
            src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            throw std::runtime_error("Vulkan failure!");
        };
    }
    return code;
}

constexpr SlangResult checkSlang(SlangResult code, slang::IBlob *diag = nullptr,
    bool panic = true, source_location src = source_location::current())
{
    if (code < 0)
    {
        const char* msg = "error";
        if (diag != nullptr)
        {
            msg = static_cast<const char*>(diag->getBufferPointer());
        }
        fprintf(stderr, "Slang assertion: %s in function %s at %s(%d)\n",
            msg, src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            throw std::runtime_error("Slang failure!");
        }
    }
    else if (diag != nullptr)
    {
        const char* msg = static_cast<const char*>(diag->getBufferPointer());
        fprintf(stderr, "Slang warning: %s in function %s at %s(%d)\n",
            msg, src.function_name(), src.file_name(), src.line()
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

} // namespace validation
} // namespace mimir