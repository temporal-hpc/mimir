#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include "color/color.hpp"

enum class DataDomain    { Domain2D, Domain3D };
enum class ResourceType  { UnstructuredBuffer, StructuredBuffer, Texture, TextureLinear };
enum class PrimitiveType { Points, Edges, Voxels };
enum class TextureFormat { Uint8, Int32, Float32, Rgba32 };

// Holds customization options for the view it is associated to, for example:
// Point color, point size, edge color, etc.
// In the future, it should only have the fields the view actually supports
struct ViewOptions
{
  color::rgba<float> color{0.f,0.f,0.f,1.f};
  float size = 1.f;
  float depth = 0.01f;
};

struct ViewParams
{
  size_t element_count = 0;
  size_t element_size = 0;
  uint3 extent = {1, 1, 1};
  DataDomain data_domain;
  ResourceType resource_type;
  PrimitiveType primitive_type;
  TextureFormat texture_format;
  ViewOptions options;
};

struct CudaView
{
  ViewParams params;

  // Interop members
  void *cuda_ptr = nullptr;
  cudaExternalMemory_t cuda_extmem = nullptr;
  VkBuffer interop_buffer = VK_NULL_HANDLE;
  VkDeviceMemory interop_memory = VK_NULL_HANDLE;

  // Auxiliary memory members
  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VkBuffer index_buffer = VK_NULL_HANDLE;
  VkDeviceMemory aux_memory = VK_NULL_HANDLE;
  VkBuffer ubo_buffer = VK_NULL_HANDLE;
  VkDeviceMemory ubo_memory = VK_NULL_HANDLE;

  // Image members
  VkFormat vk_format = VK_FORMAT_UNDEFINED;
  VkExtent3D vk_extent = {0, 0, 0};
  VkDeviceMemory img_memory = VK_NULL_HANDLE;
  VkImage image = VK_NULL_HANDLE;
  VkImageView vk_view  = VK_NULL_HANDLE;
  VkSampler vk_sampler = VK_NULL_HANDLE;

  // Cudaarrays (TODO: Move)
  cudaMipmappedArray_t cudaMipmappedImageArray = 0;
  cudaMipmappedArray_t cudaMipmappedImageArrayTemp = 0;
  cudaMipmappedArray_t cudaMipmappedImageArrayOrig = 0;
  std::vector<cudaSurfaceObject_t> surfaceObjectList, surfaceObjectListTemp;
  cudaSurfaceObject_t *d_surfaceObjectList = nullptr;
  cudaSurfaceObject_t *d_surfaceObjectListTemp = nullptr;
  cudaTextureObject_t textureObjMipMapInput = 0;
};
