#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <string> // std::string

#include "color/color.hpp"

// Specifies the number of spatial dimensions of the view
enum class DataDomain    { Domain2D, Domain3D };
// Specifies the data layout
enum class ResourceType  { UnstructuredBuffer, StructuredBuffer, Texture, TextureLinear };
// Specifies the type of primitive that will be visualized 
enum class PrimitiveType { Points, Edges, Voxels };
// Specifies the datatype stored in the texture corresponding to a view
enum class TextureFormat { Uint8, Int32, Float32, Rgba32 };

struct ShaderInfo {
  std::string filepath;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL_GRAPHICS;
};

// Holds customization options for the view it is associated to, for example:
// Point color, point size, edge color, etc.
// In the future, it should only have the fields the view actually supports
struct ViewOptions
{
  // Customizable name for the view
  std::string name;
  // Flag indicating if this view should be displayed or not
  bool visible = true;
  // Default primitive color if no per-instance color is set
  color::rgba<float> color{0.f,0.f,0.f,1.f};
  // Default primitive size if no per-instance size is set
  float size = 10.f;
  // External alternate shaders for use in this view
  std::vector<ShaderInfo> external_shaders;
  // For specializing slang shaders associated to this view
  std::vector<std::string> specializations;
  // For moving through the different slices in a 3D texture
  float depth = 0.01f;
};

struct ViewParams
{
  cudaStream_t cuda_stream = 0;
  size_t element_count = 0;
  size_t element_size = 0;
  uint3 extent = {1, 1, 1};
  DataDomain data_domain;
  ResourceType resource_type;
  PrimitiveType primitive_type;
  TextureFormat texture_format;
  ViewOptions options;
};

// Struct for storing Vulkan/Cuda interoperatibility members 
struct InteropMemory
{
  // Raw Cuda pointer which can be passed to the library user
  // for use in kernels, as per cudaMalloc
  void *cuda_ptr = nullptr;
  // Vulkan buffer handle
  VkBuffer data_buffer = VK_NULL_HANDLE;
  // Vulkan external device memory
  VkDeviceMemory memory = VK_NULL_HANDLE;  
  // Cuda external memory handle, provided by the Cuda interop API
  cudaExternalMemory_t cuda_extmem = nullptr;

  // Image members (TODO: Should be separated)
  cudaMipmappedArray_t mipmap_array = nullptr;
  VkImage image = VK_NULL_HANDLE;
};

struct CudaView
{
  ViewParams params;
  uint32_t pipeline_index = 0;
  InteropMemory _interop;

  // Auxiliary memory members
  VkBuffer vertex_buffer    = VK_NULL_HANDLE;
  VkBuffer index_buffer     = VK_NULL_HANDLE;
  VkDeviceMemory aux_memory = VK_NULL_HANDLE;
  VkBuffer ubo_buffer       = VK_NULL_HANDLE;
  VkDeviceMemory ubo_memory = VK_NULL_HANDLE;

  // Image members
  VkExtent3D vk_extent = {0, 0, 0};
  VkFormat vk_format   = VK_FORMAT_UNDEFINED;
  VkImageView vk_view  = VK_NULL_HANDLE;
  VkSampler vk_sampler = VK_NULL_HANDLE;
};
