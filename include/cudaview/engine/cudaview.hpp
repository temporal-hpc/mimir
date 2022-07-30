#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include "cudaview/engine/vk_buffer.hpp"
#include "cudaview/engine/vk_texture.hpp"

enum class DataDomain    { Domain2D, Domain3D };
enum class ResourceType  { UnstructuredBuffer, StructuredBuffer, Texture };
enum class PrimitiveType { Points, Edges, Voxels };
enum class TextureFormat { Uint8, Int32, Float32, Rgba32 };

struct ViewParams
{
  size_t element_count = 0;
  size_t element_size = 0;
  uint3 extent = {1, 1, 1};
  DataDomain data_domain;
  ResourceType resource_type;
  PrimitiveType primitive_type;
  TextureFormat texture_format;
};

struct CudaView
{
  ViewParams params;
  void *cuda_ptr = nullptr;
  cudaExternalMemory_t cuda_extmem = nullptr;
  VkBuffer interop_buffer = VK_NULL_HANDLE;
  VkDeviceMemory interop_memory = VK_NULL_HANDLE;

  VkFormat vk_format = VK_FORMAT_UNDEFINED;
  VkExtent3D vk_extent = {0, 0, 0};
  VulkanBuffer vertex_buffer;
  VulkanBuffer index_buffer;

  TextureFormat tex_format;
  VulkanBuffer implicit;
  VulkanTexture texture;
  VkImageView vk_view  = VK_NULL_HANDLE;
  VkSampler vk_sampler = VK_NULL_HANDLE;
};
