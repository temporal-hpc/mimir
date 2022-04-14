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

inline VkImageViewType getViewType(DataDomain domain)
{
  switch (domain)
  {
    case DataDomain::Domain2D: return VK_IMAGE_VIEW_TYPE_2D;
    case DataDomain::Domain3D: return VK_IMAGE_VIEW_TYPE_3D;
    default:                   return VK_IMAGE_VIEW_TYPE_1D;
  }
}

struct CudaView
{
  DataDomain data_domain;
  size_t element_count = 0;
  size_t element_size  = 0;
  void *cuda_ptr = nullptr;
  cudaExternalMemory_t cuda_extmem = nullptr;
  VkFormat vk_format = VK_FORMAT_UNDEFINED;

  CudaView();
  CudaView(size_t elem_count, size_t elem_size, DataDomain domain, DataFormat format):
    element_count{elem_count}, element_size{elem_size}, data_domain{domain},
    vk_format{getVulkanFormat(format)}
  {}

  CudaView(size_t elem_count, size_t elem_size, DataDomain domain):
    element_count{elem_count}, element_size{elem_size}, data_domain{domain},
    vk_format{VK_FORMAT_UNDEFINED}
  {}
};

struct CudaViewUnstructured : public CudaView
{
  VulkanBuffer buffer;
  UnstructuredDataType data_type;

  CudaViewUnstructured(size_t elem_count, size_t elem_size,
    DataDomain domain, UnstructuredDataType type):
  CudaView{elem_count, elem_size, domain}, data_type{type}
  {}
};

struct CudaViewStructured : public CudaView
{
  VulkanBuffer buffer;
  VulkanTexture texture;
  VkImageView vk_view  = VK_NULL_HANDLE;
  VkSampler vk_sampler = VK_NULL_HANDLE;

  CudaViewStructured(VkExtent3D extent, size_t elem_size, DataDomain domain,
    DataFormat format):
    CudaView{extent.width*extent.height*extent.depth, elem_size, domain, format}
  {}
};
