#pragma once

#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>

#include <functional>

class VulkanCudaEngine : public VulkanEngine
{
public:
  VulkanCudaEngine(int2 extent, cudaStream_t stream);
  VulkanCudaEngine();
  ~VulkanCudaEngine();
  void registerUnstructuredMemory(float *&d_cudamem, size_t element_count);
  void registerStructuredMemory(float *&d_cudamem, size_t width, size_t height);
  void registerFunction(std::function<void(void)> func, size_t iter_count);

private:
  // Cuda interop data
  size_t iteration_count, iteration_idx;
  int2 data_extent;
  std::function<void(void)> step_function;
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_wait_semaphore, cuda_signal_semaphore;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;

  float *cuda_unstructured_data;
  cudaExternalMemory_t cuda_extmem_unstructured;
  VkBuffer vk_unstructured_buffer;
  VkDeviceMemory vk_unstructured_memory;

  float *cuda_structured_data;
  cudaExternalMemory_t cuda_extmem_structured;
  VkBuffer vk_structured_buffer;
  VkDeviceMemory vk_structured_memory;

  void initVulkan();
  void drawFrame();
  void setUnstructuredRendering(VkCommandBuffer& cmd_buffer, uint32_t vertex_count);
  void updateUniformBuffer(uint32_t image_index);

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
  virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
};
