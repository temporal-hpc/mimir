#pragma once

#include "cudaview/vk_types.hpp"

namespace utils
{

uint32_t findMemoryType(VkPhysicalDevice ph_device, uint32_t type_filter,
  VkMemoryPropertyFlags properties
);
void listAvailableExtensions();

} // namespace utils
