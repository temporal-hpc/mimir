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

VulkanCudaEngine::VulkanCudaEngine():
  vk_data_buffer(VK_NULL_HANDLE),
  vk_data_memory(VK_NULL_HANDLE),
  stream(0),
  cuda_timeline_semaphore(nullptr),
  cuda_vert_memory(nullptr),
  cuda_raw_data(nullptr),
  element_count(0)
{}

VulkanCudaEngine::~VulkanCudaEngine()
{
  // Make sure there is no pending work before cleanup starts
  checkCuda(cudaStreamSynchronize(stream));

  if (vk_timeline_semaphore != VK_NULL_HANDLE)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  }
  if (vk_data_buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, vk_data_buffer, nullptr);
  }
  if (vk_data_memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, vk_data_memory, nullptr);
  }
  if (cuda_raw_data != nullptr)
  {
    checkCuda(cudaDestroyExternalMemory(cuda_vert_memory));
  }
}

float *VulkanCudaEngine::allocateDeviceMemory(size_t p_elements)
{
  element_count = p_elements;
  createExternalBuffer(sizeof(float2) * element_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    vk_data_buffer, vk_data_memory
  );
  importCudaExternalMemory((void**)&cuda_raw_data, cuda_vert_memory,
    vk_data_memory, sizeof(*cuda_raw_data) * element_count
  );
  return cuda_raw_data;
}

void VulkanCudaEngine::setUnstructuredRendering(VkCommandBuffer& cmd_buffer,
  uint32_t vertex_count)
{
  VkBuffer vertex_buffers[] = { vk_data_buffer };
  VkDeviceSize offsets[] = { 0 };
  auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
  vkCmdBindVertexBuffers(cmd_buffer, 0, binding_count, vertex_buffers, offsets);
  vkCmdDraw(cmd_buffer, vertex_count, 1, 0, 0);
}

void VulkanCudaEngine::getVertexDescriptions(
  VkVertexInputBindingDescription& bind_desc,
  VkVertexInputAttributeDescription& attr_desc)
{
  bind_desc.binding = 0;
  bind_desc.stride = sizeof(float2);
  bind_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.binding = 0;
  attr_desc.location = 0;
  attr_desc.format = VK_FORMAT_R32G32_SFLOAT;
  attr_desc.offset = 0;
}

void VulkanCudaEngine::getAssemblyStateInfo(
  VkPipelineInputAssemblyStateCreateInfo &info)
{
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  info.primitiveRestartEnable = VK_FALSE;
}

void VulkanCudaEngine::initApplication()
{
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);
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
  //return VulkanEngine::getRequiredExtensions();
  //std::vector<const char*> extensions;
  auto extensions = VulkanEngine::getRequiredExtensions();
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  return extensions;
}

std::vector<const char*> VulkanCudaEngine::getRequiredDeviceExtensions() const
{
  //return VulkanEngine::getRequiredDeviceExtensions();
  //std::vector<const char*> extensions;
  auto extensions = VulkanEngine::getRequiredDeviceExtensions();
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  return extensions;
}
