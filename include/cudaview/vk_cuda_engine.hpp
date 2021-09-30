#pragma once

#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>

class VulkanCudaEngine : public VulkanEngine
{
public:
  ~VulkanCudaEngine();
  float *allocateDeviceMemory(size_t element_count);

private:
  // Vulkan interop data
  VkBuffer vk_data_buffer = VK_NULL_HANDLE;
  VkDeviceMemory vk_data_memory = VK_NULL_HANDLE;
  VkSemaphore vk_timeline_semaphore = VK_NULL_HANDLE;
  VkSemaphore vk_wait_semaphore = VK_NULL_HANDLE;
  VkSemaphore vk_signal_semaphore = VK_NULL_HANDLE;

  // Cuda interop data
  cudaStream_t stream = 0;
  cudaExternalSemaphore_t cuda_wait_semaphore = nullptr;
  cudaExternalSemaphore_t cuda_signal_semaphore = nullptr;
  cudaExternalSemaphore_t cuda_timeline_semaphore = nullptr;
  cudaExternalMemory_t cuda_vert_memory;
  float *cuda_raw_data = nullptr;
  size_t element_count = 0;

  void initApplication();
  void drawFrame();
  void setUnstructuredRendering(VkCommandBuffer& cmd_buffer, uint32_t vertex_count);

  void getVertexDescriptions(VkVertexInputBindingDescription& bind_desc,
    VkVertexInputAttributeDescription& attr_desc
  );
  void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo &info);

  // Interop semaphore handling functions
  void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;

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
