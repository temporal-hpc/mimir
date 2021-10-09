#pragma once

#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>

#include <functional>

class VulkanCudaEngine : public VulkanEngine
{
public:
  VulkanCudaEngine(size_t data_size, cudaStream_t stream);
  VulkanCudaEngine();
  ~VulkanCudaEngine();
  void registerDeviceMemory(float *&d_cudamem);
  void registerFunction(std::function<void(void)> func, size_t iter_count);

private:
  // Vulkan interop data
  VkBuffer vk_data_buffer;
  VkDeviceMemory vk_data_memory;

  // Cuda interop data
  cudaStream_t stream;
  float *cuda_raw_data;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;
  cudaExternalSemaphore_t cuda_wait_semaphore, cuda_signal_semaphore;
  cudaExternalMemory_t cuda_vert_memory;
  std::function<void(void)> step_function;
  size_t iteration_count, iteration_idx;

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
  virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
};
