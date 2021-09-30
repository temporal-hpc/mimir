#include "cudaview/vk_cuda_engine.hpp"
#include "validation.hpp"

#include <experimental/source_location>

using source_location = std::experimental::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
  source_location src = source_location::current())
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA assertion: %s on function %s at %s(%d)\n",
      cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
    );
    if (panic)
    {
      throw std::runtime_error("CUDA failure!");
    }
  }
}

VulkanCudaEngine::~VulkanCudaEngine()
{
  // Make sure there is no pending work before cleanup starts
  checkCuda(cudaStreamSynchronize(stream));

  checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  vkDestroySemaphore(device, vk_timeline_semaphore, nullptr);

  checkCuda(cudaDestroyExternalSemaphore(cuda_signal_semaphore));
  vkDestroySemaphore(device, vk_signal_semaphore, nullptr);

  checkCuda(cudaDestroyExternalSemaphore(cuda_wait_semaphore));
  vkDestroySemaphore(device, vk_wait_semaphore, nullptr);

  vkDestroyBuffer(device, vk_data_buffer, nullptr);
  vkFreeMemory(device, vk_data_memory, nullptr);
  checkCuda(cudaDestroyExternalMemory(cuda_vert_memory));
}

void VulkanCudaEngine::fillRenderingCommandBuffer(VkCommandBuffer& cmd_buffer)
{
  VkBuffer vertex_buffers[] = { vk_data_buffer };
  VkDeviceSize offsets[] = { 0 };
  auto binding_count = 1; //sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
  vkCmdBindVertexBuffers(cmd_buffer, 0, binding_count, vertex_buffers, offsets);
  vkCmdDraw(cmd_buffer, data_size, 1, 0, 0);
  // alternatively, use vkCmdBindIndexBuffer(...)
}

void VulkanCudaEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
  std::vector<VkPipelineStageFlags>& wait_stages) const
{
  if (current_frame != 0)
  {
    wait.push_back(vk_wait_semaphore);
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
}

void VulkanCudaEngine::getSignalFrameSemaphores(
  std::vector<VkSemaphore>& signal) const
{
  signal.push_back(vk_signal_semaphore);
}

void VulkanCudaEngine::initInterop(size_t vertex_count)
{
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  data_size = vertex_count;
}

void VulkanCudaEngine::initApplication()
{
  createExternalBuffer(sizeof(float2) * data_size,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    vk_data_buffer, vk_data_memory
  );
  importCudaExternalMemory((void**)&cuda_raw_data, cuda_vert_memory,
    vk_data_memory, sizeof(*cuda_raw_data) * data_size
  );

  createExternalSemaphore(vk_timeline_semaphore);
  importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);

  createExternalSemaphore(vk_signal_semaphore);
  importCudaExternalSemaphore(cuda_signal_semaphore, vk_signal_semaphore);

  createExternalSemaphore(vk_wait_semaphore);
  importCudaExternalSemaphore(cuda_wait_semaphore, vk_wait_semaphore);
}

void VulkanCudaEngine::importCudaExternalMemory(void **cuda_ptr,
  cudaExternalMemory_t& cuda_mem, VkDeviceMemory& vk_mem, VkDeviceSize size)
{
  cudaExternalMemoryHandleDesc extmem_desc{};
  extmem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  extmem_desc.size = size;
  extmem_desc.handle.fd = (int)(uintptr_t)getMemHandle(
    vk_mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
  );

  checkCuda(cudaImportExternalMemory(&cuda_mem, &extmem_desc));

  cudaExternalMemoryBufferDesc buffer_desc{};
  buffer_desc.offset = 0;
  buffer_desc.size = size;
  buffer_desc.flags = 0;

  checkCuda(cudaExternalMemoryGetMappedBuffer(cuda_ptr, cuda_mem, &buffer_desc));
}

void VulkanCudaEngine::importCudaExternalSemaphore(
  cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem)
{
  cudaExternalSemaphoreHandleDesc sem_desc{};
  sem_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  sem_desc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(
    vk_sem, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
  );
  sem_desc.flags = 0;
  checkCuda(cudaImportExternalSemaphore(&cuda_sem, &sem_desc));
}

void VulkanCudaEngine::drawFrame()
{
  VulkanEngine::drawFrame();

  cudaExternalSemaphoreWaitParams wait_params{};
  wait_params.flags = 0;
  wait_params.params.fence.value = 1;

  cudaExternalSemaphoreSignalParams signal_params{};
  signal_params.flags = 0;
  signal_params.params.fence.value = 0;

  // Wait for Vulkan to complete its work
  checkCuda(cudaWaitExternalSemaphoresAsync(
    &cuda_timeline_semaphore, &wait_params, 1, stream)
  );
  // TODO: stepSimulation()
  // Signal Vulkan to continue with the updated buffers
  checkCuda(cudaSignalExternalSemaphoresAsync(
    &cuda_timeline_semaphore, &signal_params, 1, stream)
  );
}

std::vector<const char*> VulkanCudaEngine::getRequiredExtensions() const
{
  std::vector<const char*> extensions;
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  return extensions;
}

std::vector<const char*> VulkanCudaEngine::getRequiredDeviceExtensions() const
{
  std::vector<const char*> extensions;
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  return extensions;
}
