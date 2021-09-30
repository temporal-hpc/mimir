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

void VulkanCudaEngine::initInterop(size_t vertex_count)
{
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  createExternalBuffer(sizeof(float2) * vertex_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    vk_data_buffer, vk_data_memory
  );

  importCudaExternalMemory((void**)&cuda_raw_data, cuda_vert_memory,
    vk_data_memory, sizeof(*cuda_raw_data) * vertex_count
  );

  createExternalSemaphore(vk_timeline_semaphore);
  importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);
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

void VulkanCudaEngine::createExternalSemaphore(VkSemaphore& semaphore)
{
  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkExportSemaphoreCreateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_info.pNext = nullptr;

  VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.pNext = nullptr;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_info.initialValue = 0;
  export_info.pNext = &timeline_info;
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  semaphore_info.pNext = &export_info;

  validation::checkVulkan(
    vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore)
  );
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

void *VulkanCudaEngine::getSemaphoreHandle(VkSemaphore semaphore,
  VkExternalSemaphoreHandleTypeFlagBits handle_type)
{
  int fd;
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.semaphore = semaphore;
  fd_info.handleType = handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphore;
  fpGetSemaphore = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
    device, "PFN_vkGetSemaphoreFdKHR"
  );
  if (!fpGetSemaphore)
  {
    throw std::runtime_error("Failed to retrieve semaphore handle!");
  }
  validation::checkVulkan(fpGetSemaphore(device, &fd_info, &fd));

  return (void*)(uintptr_t)fd;
}

void *VulkanCudaEngine::getMemHandle(VkDeviceMemory memory,
  VkExternalMemoryHandleTypeFlagBits handle_type)
{
  int fd = -1;

  VkMemoryGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.memory = memory;
  fd_info.handleType = handle_type;

  auto fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(
    device, "vkGetMemoryFdKHR"
  );
  if (!fpGetMemoryFdKHR)
  {
    throw std::runtime_error("Failed to retrieve function!");
  }
  if (fpGetMemoryFdKHR(device, &fd_info, &fd) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return (void*)(uintptr_t)fd;
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
