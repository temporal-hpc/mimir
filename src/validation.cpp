#include "internal/validation.hpp"

#include <cstring> // strcmp

namespace validation
{

// Setup debug messenger callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
  VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
  VkDebugUtilsMessageTypeFlagsEXT message_type,
  const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
  void *p_user_data)
{
  fprintf(stderr, "Validation layer: %s\n", p_callback_data->pMessage);
  return VK_FALSE;
}

std::string getVulkanErrorString(VkResult code)
{
  switch (code)
  {
#define STR(r) case VK_ ##r: return #r
    STR(NOT_READY);
		STR(TIMEOUT);
		STR(EVENT_SET);
		STR(EVENT_RESET);
		STR(INCOMPLETE);
		STR(ERROR_OUT_OF_HOST_MEMORY);
		STR(ERROR_OUT_OF_DEVICE_MEMORY);
		STR(ERROR_INITIALIZATION_FAILED);
		STR(ERROR_DEVICE_LOST);
		STR(ERROR_MEMORY_MAP_FAILED);
		STR(ERROR_LAYER_NOT_PRESENT);
		STR(ERROR_EXTENSION_NOT_PRESENT);
		STR(ERROR_FEATURE_NOT_PRESENT);
		STR(ERROR_INCOMPATIBLE_DRIVER);
		STR(ERROR_TOO_MANY_OBJECTS);
		STR(ERROR_FORMAT_NOT_SUPPORTED);
		STR(ERROR_SURFACE_LOST_KHR);
		STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
		STR(SUBOPTIMAL_KHR);
		STR(ERROR_OUT_OF_DATE_KHR);
		STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
		STR(ERROR_VALIDATION_FAILED_EXT);
		STR(ERROR_INVALID_SHADER_NV);
#undef STR
		default:
			return "UNKNOWN_ERROR";
  }
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
  const VkDebugUtilsMessengerCreateInfoEXT *p_create_info,
  const VkAllocationCallbacks *p_allocator,
  VkDebugUtilsMessengerEXT *p_debug_messenger)
{
  // Lookup address of debug messenger extension function
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
  {
    return func(instance, p_create_info, p_allocator, p_debug_messenger);
  }
  else // Function could not be loaded
  {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
  VkDebugUtilsMessengerEXT debug_messenger,
  const VkAllocationCallbacks *p_allocator
)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
    vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
  {
    return func(instance, debug_messenger, p_allocator);
  }
}

// TODO: Should it be in vk_initializers?
VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo()
{
  VkDebugUtilsMessengerCreateInfoEXT info{};
  info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  info.messageSeverity =
    //VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  info.messageType =
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  info.pfnUserCallback = debugCallback;
  info.pUserData = nullptr; // optional
  return info;
}

bool checkValidationLayerSupport()
{
  // List all available layers
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  // Check if all of the validation layers are available
  for (const auto layerName : layers)
  {
    bool layer_found = false;
    for (const auto& layer_properties : available_layers)
    {
      if (strcmp(layerName, layer_properties.layerName) == 0)
      {
        layer_found = true;
        break;
      }
    }
    if (!layer_found)
    {
      return false;
    }
  }
  return true;
}

} // namespace validation
