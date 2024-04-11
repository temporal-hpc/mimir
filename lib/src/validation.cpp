#include <mimir/validation.hpp>

#include <cstring> // strcmp

namespace mimir::validation
{

// Converts Vulkan message severity flags into a string for logging
const char* getVulkanSeverityString(VkDebugUtilsMessageSeverityFlagBitsEXT flag)
{
    switch (flag)
    {
#define STR(r) case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ ## r ## _BIT_EXT: return #r
        STR(VERBOSE);
        STR(INFO);
        STR(WARNING);
        STR(ERROR);
#undef STR
        default: return "UNKNOWN";
    }
}

// Converts Vulkan message type flags into a string for logging
const char* getVulkanMessageType(VkDebugUtilsMessageTypeFlagsEXT type)
{
    switch (type)
    {
#define STR(r) case VK_DEBUG_UTILS_MESSAGE_TYPE_ ## r ## _BIT_EXT: return #r
        STR(GENERAL);
        STR(VALIDATION);
        STR(PERFORMANCE);
        // STR(DEVICE_ADDRESS_BINDING); // TODO: May require extension, investigate
#undef STR
        default: return "UNKNOWN";
    }
}

// Setup debug messenger callback. Currently it just converts debug data into a string for logging
// TODO: Maybe do something with p_user_data, like passing a pointer
// to the engine class in order to get access to its fields
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
    VkDebugUtilsMessageTypeFlagsEXT msg_type,
    const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
    [[maybe_unused]] void *p_user_data)
{
    auto severity = getVulkanSeverityString(msg_severity);
    auto type = getVulkanMessageType(msg_type);
    fprintf(stderr, "[%s] %s: %s\n", severity, type, p_callback_data->pMessage);
    return VK_FALSE;
}

// Converts Vulkan result codes into strings
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
        default: return "UNKNOWN_ERROR";
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
    const VkAllocationCallbacks *p_allocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, debug_messenger, p_allocator);
    }
}

VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo()
{
    VkDebugUtilsMessengerCreateInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity =
            //VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = nullptr, // optional
    };
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

} // namespace mimir::validation