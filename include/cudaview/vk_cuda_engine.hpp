#pragma once

#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>

#include <functional>

enum class DataFormat
{
  Float32,
  Rgba32
};

struct MappedUnstructuredMemory
{
  void *cuda_ptr;
  cudaExternalMemory_t cuda_extmem;
  size_t element_count;
  size_t element_size;
  VkFormat vk_format;
  VkBuffer vk_buffer;
  VkDeviceMemory vk_memory;
};

struct MappedStructuredMemory
{
  void *cuda_ptr;
  cudaExternalMemory_t cuda_extmem;
  size_t element_count;
  size_t element_size;
  VkFormat vk_format;
  VkBuffer vk_buffer;
  VkDeviceMemory vk_memory;
  VkImage vk_image;
  VkImageView vk_view;
  std::array<ulong, 3> extent;
};

class VulkanCudaEngine : public VulkanEngine
{
public:
  VulkanCudaEngine(int2 extent, cudaStream_t stream = 0);
  VulkanCudaEngine();
  ~VulkanCudaEngine();
  void registerUnstructuredMemory(void **ptr_devmem, size_t elem_count, size_t elem_size);
  void registerStructuredMemory(void **ptr_devmem, size_t width, size_t height,
    size_t elem_size, DataFormat format
  );
  void registerFunction(std::function<void(void)> func, size_t iter_count);

  void display();
  void displayAsync();
  void mainLoopThreaded();
  void prepareWindow();
  void updateWindow();

private:
  // Cuda interop data
  size_t iteration_count, iteration_idx;
  int2 data_extent;
  std::function<void(void)> step_function;
  cudaStream_t stream;
  cudaExternalSemaphore_t cuda_wait_semaphore, cuda_signal_semaphore;
  //cudaExternalSemaphore_t cuda_timeline_semaphore;

  std::vector<MappedStructuredMemory> structured_buffers;
  std::vector<MappedUnstructuredMemory> unstructured_buffers;

  void *cuda_unstructured_data;
  cudaExternalMemory_t cuda_extmem_unstructured;
  VkBuffer vk_unstructured_buffer;
  VkDeviceMemory vk_unstructured_memory;

  void initVulkan();
  void cudaSemaphoreSignal();
  void cudaSemaphoreWait();
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

  void updateDescriptorsUnstructured();
  void updateDescriptorsStructured();

  // Handle additional extensions required by CUDA interop
  std::vector<const char*> getRequiredExtensions() const;
  std::vector<const char*> getRequiredDeviceExtensions() const;
  virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
    std::vector<VkPipelineStageFlags>& wait_stages) const;
  virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
};
