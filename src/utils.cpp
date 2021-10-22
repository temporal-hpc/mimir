#include "utils.hpp"

#include <iostream>

namespace utils
{

uint32_t findMemoryType(VkPhysicalDevice ph_device, uint32_t type_filter,
  VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(ph_device, &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
  {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }
  return ~0;
}

void listAvailableExtensions()
{
  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> available_exts(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_exts.data());

  std::cout << "Available extensions:\n";
  for (const auto& extension : available_exts)
  {
    std::cout << '\t' << extension.extensionName << '\n';
  }
}

} // namespace utils
