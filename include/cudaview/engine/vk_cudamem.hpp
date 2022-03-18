#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include "cudaview/engine/vk_buffer.hpp"
#include "cudaview/engine/vk_texture.hpp"

enum class DataFormat { Uint8, Int32, Float32, Rgba32 };
enum class DataDomain { Domain2D, Domain3D };
enum class UnstructuredDataType { Points, Edges };

inline VkFormat getVulkanFormat(DataFormat format)
{
  switch (format)
  {
    case DataFormat::Uint8:   return VK_FORMAT_R8_UNORM;
    case DataFormat::Int32:   return VK_FORMAT_R32_SINT;
    case DataFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case DataFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                  return VK_FORMAT_UNDEFINED;
  }
}

inline VkImageType getImageDimensions(DataDomain domain)
{
  switch (domain)
  {
    case DataDomain::Domain2D: return VK_IMAGE_TYPE_2D;
    case DataDomain::Domain3D: return VK_IMAGE_TYPE_3D;
    default:                   return VK_IMAGE_TYPE_1D;
  }
}

struct CudaMappedMemory
{
  size_t element_count = 0;
  size_t element_size  = 0;
  void *cuda_ptr = nullptr;
  cudaExternalMemory_t cuda_extmem = nullptr;
  VkFormat vk_format = VK_FORMAT_UNDEFINED;
  DataDomain data_domain;

  CudaMappedMemory();
  CudaMappedMemory(size_t elem_count, size_t elem_size, DataDomain domain, DataFormat format):
    element_count{elem_count}, element_size{elem_size}, data_domain{domain},
    vk_format{getVulkanFormat(format)}
  {}

  CudaMappedMemory(size_t elem_count, size_t elem_size, DataDomain domain):
    element_count{elem_count}, element_size{elem_size}, data_domain{domain},
    vk_format{VK_FORMAT_UNDEFINED}
  {}
};

struct MappedUnstructuredMemory : public CudaMappedMemory
{
  VulkanBuffer buffer;
  UnstructuredDataType data_type;

  MappedUnstructuredMemory(size_t elem_count, size_t elem_size,
    DataDomain domain, UnstructuredDataType type):
  CudaMappedMemory{elem_count, elem_size, domain}, data_type{type}
  {}
};

struct MappedStructuredMemory : public CudaMappedMemory
{
  VulkanTexture texture;
  VkImageView vk_view;

  MappedStructuredMemory(VkExtent3D extent, size_t elem_size, DataDomain domain,
    DataFormat format):
    CudaMappedMemory{extent.width*extent.height*extent.depth, elem_size, domain, format}
  {}
};
