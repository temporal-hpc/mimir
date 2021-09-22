#pragma once

#include "cudaview/vk_types.hpp"

// Check if validation layers should be enabled
#ifdef NDEBUG
  const bool enable_validation_layers = false;
#else
  const bool enable_validation_layers = true;
#endif

// Validation layers to enable
const std::vector<const char*> validation_layers = {
  "VK_LAYER_KHRONOS_validation"
};
// Required device extensions
const std::vector<const char*> device_extensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
