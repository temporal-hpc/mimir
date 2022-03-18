#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include "cudaview/engine/vk_buffer.hpp"
#include "cudaview/engine/vk_texture.hpp"

enum class DataFormat { Float32, Rgba32 };
enum class DataDomain { Domain2D, Domain3D };
enum class UnstructuredDataType { Points, Edges };

inline VkFormat getVulkanFormat(DataFormat format)
{
  switch (format)
  {
    case DataFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case DataFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                  return VK_FORMAT_UNDEFINED;
  }
}

struct CudaMappedMemory
{
  size_t element_count = 0;
  size_t element_size  = 0;
  void *cuda_ptr = nullptr;
  cudaExternalMemory_t cuda_extmem = nullptr;
  VkFormat vk_format = VK_FORMAT_UNDEFINED;

  CudaMappedMemory();
  CudaMappedMemory(size_t elem_count, size_t elem_size, DataFormat format):
    element_count{elem_count}, element_size{elem_size}, vk_format{getVulkanFormat(format)}
  {}

  CudaMappedMemory(size_t elem_count, size_t elem_size):
    element_count{elem_count}, element_size{elem_size}, vk_format{VK_FORMAT_UNDEFINED}
  {}
};

struct MappedUnstructuredMemory : public CudaMappedMemory
{
  VulkanBuffer buffer;
  DataDomain data_domain;
  UnstructuredDataType data_type;

  MappedUnstructuredMemory(size_t elem_count, size_t elem_size,
    UnstructuredDataType type, DataDomain domain):
  CudaMappedMemory{elem_count, elem_size}, data_type{type}, data_domain{domain}
  {}
};

struct MappedStructuredMemory : public CudaMappedMemory
{
  VulkanTexture texture;
  VkImageView vk_view;

  MappedStructuredMemory(VkExtent3D extent, size_t elem_size, DataFormat format):
    CudaMappedMemory{extent.width*extent.height*extent.depth, elem_size, format}
  {}
};
