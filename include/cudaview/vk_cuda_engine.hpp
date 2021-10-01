#pragma once

#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>

#include <functional>

class VulkanCudaEngine : public VulkanEngine
{
public:
  VulkanCudaEngine(size_t data_size);
  VulkanCudaEngine();
  ~VulkanCudaEngine();
  float *getDeviceMemory();
  void registerFunction(std::function<void(void)> func);

private:
  // Vulkan interop data
  VkBuffer vk_data_buffer;
  VkDeviceMemory vk_data_memory;

  // Cuda interop data
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_timeline_semaphore;
  cudaExternalMemory_t cuda_vert_memory;
  float *cuda_raw_data;
  std::function<void(void)> step_function;

  void initApplication();
  void drawFrame();
  void setUnstructuredRendering(VkCommandBuffer& cmd_buffer, uint32_t vertex_count);

  void getVertexDescriptions(
    std::vector<VkVertexInputBindingDescription>& bind_desc,
    std::vector<VkVertexInputAttributeDescription>& attr_desc
  );
  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);

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
