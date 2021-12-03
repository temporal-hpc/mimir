#pragma once

#include <experimental/source_location>

#include "cudaview/vk_engine.hpp"

namespace validation
{

// Check if validation layers should be enabled
#ifdef NDEBUG
  constexpr bool enable_layers = false;
#else
  constexpr bool enable_layers = true;
#endif

using source_location = std::experimental::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
  source_location src = source_location::current())
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA assertion: %s on function %s at %s(%d)\n",
      cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
    );
    if (panic)
    {
      throw std::runtime_error("CUDA failure!");
    }
  }
}

// Validation layers to enable
const std::vector<const char*> layers = {
  "VK_LAYER_KHRONOS_validation"
};

std::string getVulkanErrorString(VkResult code);

constexpr VkResult checkVulkan(VkResult code, bool panic = true,
  source_location src = source_location::current())
{
  if (code != VK_SUCCESS)
  {
    fprintf(stderr, "Vulkan assertion: %s on function %s at %s(%d)\n",
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
