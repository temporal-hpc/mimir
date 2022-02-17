#include "cudaview/engine/vk_cudamem.hpp"

MappedUnstructuredMemory newUnstructuredMemory(size_t elem_count,
  size_t elem_size, UnstructuredDataType type, DataDomain domain)
{
  MappedUnstructuredMemory mapped{};
  mapped.element_count = elem_count;
  mapped.element_size  = elem_size;
  mapped.data_type     = type;
  mapped.data_domain   = domain;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = VK_FORMAT_UNDEFINED; //getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;

  return mapped;
}

MappedStructuredMemory newStructuredMemory(
  size_t width, size_t height, size_t elem_size, DataFormat format)
{
  MappedStructuredMemory mapped{};
  mapped.element_count = width * height;
  mapped.element_size  = elem_size;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;
  mapped.vk_image      = VK_NULL_HANDLE;
  mapped.vk_view       = VK_NULL_HANDLE;

  return mapped;
}

VkFormat getVulkanFormat(DataFormat format)
{
  switch (format)
  {
    case DataFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case DataFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                  return VK_FORMAT_UNDEFINED;
  }
}
