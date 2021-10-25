#include "cudaview/vk_cuda_engine.hpp"
#include "validation.hpp"

#include <experimental/source_location>
#include <iostream>

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

VulkanCudaEngine::VulkanCudaEngine(cudaStream_t cuda_stream):
  VulkanEngine(),
  iteration_count(0),
  iteration_idx(0),
  stream(cuda_stream),
  //cuda_timeline_semaphore(nullptr),
  cuda_wait_semaphore(nullptr),
  cuda_signal_semaphore(nullptr),

  cuda_unstructured_data(nullptr),
  cuda_extmem_unstructured(nullptr),
  vk_unstructured_buffer(VK_NULL_HANDLE),
  vk_unstructured_memory(VK_NULL_HANDLE),

  cuda_structured_data(nullptr),
  cuda_extmem_structured(nullptr),
  vk_structured_buffer(VK_NULL_HANDLE),
  vk_structured_memory(VK_NULL_HANDLE)
{}

VulkanCudaEngine::~VulkanCudaEngine()
{
  vkDeviceWaitIdle(device);

  if (stream != nullptr)
  {
    checkCuda(cudaStreamSynchronize(stream));
  }
  /*if (cuda_timeline_semaphore != nullptr)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  }*/
  if (cuda_wait_semaphore != nullptr)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_wait_semaphore));
  }
  if (cuda_signal_semaphore != nullptr)
  {
    checkCuda(cudaDestroyExternalSemaphore(cuda_signal_semaphore));
  }

  if (cuda_unstructured_data != nullptr)
  {
    checkCuda(cudaDestroyExternalMemory(cuda_extmem_unstructured));
  }
  if (vk_unstructured_buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, vk_unstructured_buffer, nullptr);
  }
  if (vk_unstructured_memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, vk_unstructured_memory, nullptr);
  }

  if (cuda_structured_data != nullptr)
  {
    checkCuda(cudaDestroyExternalMemory(cuda_extmem_structured));
  }
  if (vk_structured_buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, vk_structured_buffer, nullptr);
  }
  if (vk_structured_memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(device, vk_structured_memory, nullptr);
  }
}

void VulkanCudaEngine::registerUnstructuredMemory(float *&d_cudamem,
  size_t elem_count)
{
  element_count = elem_count;
  // Init unstructured memory
  createExternalBuffer(sizeof(float2) * elem_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    vk_unstructured_buffer, vk_unstructured_memory
  );
  importCudaExternalMemory((void**)&cuda_unstructured_data, cuda_extmem_unstructured,
    vk_unstructured_memory, sizeof(*cuda_unstructured_data) * elem_count
  );

  toggleRenderingMode("unstructured");
  d_cudamem = cuda_unstructured_data;
}

void VulkanCudaEngine::registerStructuredMemory(unsigned char *&d_cudamem,
  size_t width, size_t height)
{
  // Init structured memory
  createExternalBuffer(sizeof(uchar4) * width * height,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    vk_structured_buffer, vk_structured_memory
  );
  importCudaExternalMemory((void**)&cuda_structured_data, cuda_extmem_structured,
    vk_structured_memory, sizeof(*cuda_structured_data) * width * height * 4
  );

  toggleRenderingMode("structured");
  d_cudamem = cuda_structured_data;
}

void VulkanCudaEngine::registerFunction(std::function<void(void)> func,
  size_t iter_count)
{
  step_function = func;
  iteration_count = iter_count;
}

void VulkanCudaEngine::initVulkan()
{
  VulkanEngine::initVulkan();

  //createExternalSemaphore(vk_timeline_semaphore);
  //importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);

  createExternalSemaphore(vk_wait_semaphore);
  importCudaExternalSemaphore(cuda_signal_semaphore, vk_wait_semaphore);

  createExternalSemaphore(vk_signal_semaphore);
  importCudaExternalSemaphore(cuda_wait_semaphore, vk_signal_semaphore);
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
  //sem_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  sem_desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
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
  wait_params.params.fence.value = 0;

  cudaExternalSemaphoreSignalParams signal_params{};
  signal_params.flags = 0;
  signal_params.params.fence.value = 0;

  // Wait for Vulkan to complete its work
  checkCuda(cudaWaitExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_wait_semaphore, &wait_params, 1, stream)
  );
  if (iteration_idx <= iteration_count)
  {
    // Advance the simulation
    step_function();
    iteration_idx++;
  }
  // Signal Vulkan to continue with the updated buffers
  checkCuda(cudaSignalExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_signal_semaphore, &signal_params, 1, stream)
  );
}

std::vector<const char*> VulkanCudaEngine::getRequiredExtensions() const
{
  auto extensions = VulkanEngine::getRequiredExtensions();
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  return extensions;
}

std::vector<const char*> VulkanCudaEngine::getRequiredDeviceExtensions() const
{
  auto extensions = VulkanEngine::getRequiredDeviceExtensions();
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  return extensions;
}

void VulkanCudaEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
  std::vector<VkPipelineStageFlags>& wait_stages) const
{
  // Wait semaphore has not been initialized on the first frame
  if (current_frame != 0)
  {
    // Vulkan waits until Cuda is done with the display buffer before rendering
    wait.push_back(vk_wait_semaphore);
    // Cuda will wait until all pipeline commands are complete
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
}

void VulkanCudaEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
  // Vulkan will signal to this semaphore once the vertex ready is ready
  // for Cuda to process
  signal.push_back(vk_signal_semaphore);
}

void VulkanCudaEngine::setUnstructuredRendering(VkCommandBuffer& cmd_buffer,
  uint32_t vertex_count)
{
  VkBuffer vertex_buffers[] = { vk_unstructured_buffer };
  VkDeviceSize offsets[] = { 0 };
  auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
  vkCmdBindVertexBuffers(cmd_buffer, 0, binding_count, vertex_buffers, offsets);
  vkCmdDraw(cmd_buffer, vertex_count, 1, 0, 0);
}

void VulkanCudaEngine::getVertexDescriptions(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(float2);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(1);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
  attr_desc[0].offset = 0;
}

void VulkanCudaEngine::getAssemblyStateInfo(
  VkPipelineInputAssemblyStateCreateInfo& info)
{
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  info.primitiveRestartEnable = VK_FALSE;
}
