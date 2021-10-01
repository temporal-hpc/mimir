#pragma once

#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>

class VulkanCudaEngine : public VulkanEngine
{
public:
  VulkanCudaEngine();
  ~VulkanCudaEngine();
  float *allocateDeviceMemory(size_t element_count);

private:
  // Vulkan interop data
  VkBuffer vk_data_buffer;
  VkDeviceMemory vk_data_memory;

  // Cuda interop data
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_timeline_semaphore;
  cudaExternalMemory_t cuda_vert_memory;
  float *cuda_raw_data;
  size_t element_count;

  void initApplication();
  void drawFrame();
  void setUnstructuredRendering(VkCommandBuffer& cmd_buffer, uint32_t vertex_count);

  void getVertexDescriptions(VkVertexInputBindingDescription& bind_desc,
    VkVertexInputAttributeDescription& attr_desc
  );
  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo &info);

  // Interop import functions
  void importCudaExternalMemory(void **cuda_ptr,
    cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size
  );
  void importCudaExternalSemaphore(
    cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem
  );

  // Handle additional extensions required by CUDA interop
  std::vector<const char*> getRequiredExtensions() const;
  std::vector<const char*> getRequiredDeviceExtensions() const;
};
