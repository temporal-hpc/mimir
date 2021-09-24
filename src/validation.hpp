#pragma once

#include <experimental/source_location>

#include "cudaview/vk_engine.hpp"

namespace validation
{

// Check if validation layers should be enabled
#ifdef NDEBUG
  const bool enable_validation_layers = false;
#else
  const bool enable_validation_layers = true;
#endif

// Validation layers to enable
const std::vector<const char*> layers = {
  "VK_LAYER_KHRONOS_validation"
};

constexpr void checkVulkan(VkResult code, bool panic = true,
  std::experimental::source_location src = std::experimental::source_location::current())
{
  if (code != VK_SUCCESS)
  {
    fprintf(stderr, "Vulkan assertion: %d on function %s at %s(%d)\n",
      code, src.function_name(), src.file_name(), src.line()
    );
    if (panic) exit(code);
  }
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

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &create_info);

bool checkValidationLayerSupport();

} // namespace validation
