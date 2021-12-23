#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

enum class DataFormat { Float32, Rgba32 };
enum class DataDomain { Domain2D, Domain3D };
enum class UnstructuredDataType { Points, Edges };

struct MappedMemory {
  size_t element_count;
  size_t element_size;
  void *cuda_ptr;
  cudaExternalMemory_t cuda_extmem;
  VkFormat vk_format;
  VkBuffer vk_buffer;
  VkDeviceMemory vk_memory;
};

struct MappedUnstructuredMemory : MappedMemory
{
  DataDomain data_domain;
  UnstructuredDataType data_type;
};

struct MappedStructuredMemory : MappedMemory
{
  VkImage vk_image;
  VkImageView vk_view;
};

MappedUnstructuredMemory newUnstructuredMemory(size_t elem_count,
  size_t elem_size, UnstructuredDataType type, DataDomain domain
);
MappedStructuredMemory newStructuredMemory(
  size_t width, size_t height, size_t elem_size, DataFormat format
);
VkFormat getVulkanFormat(DataFormat format);
