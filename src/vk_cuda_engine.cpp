#include "cudaview/vk_cuda_engine.hpp"
#include "validation.hpp"

#include <experimental/source_location>
#include <iostream>
#include <cstring>

#include "cudaview/vk_types.hpp"

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

VkFormat getVulkanFormat(DataFormat format)
{
  switch (format)
  {
    case DataFormat::Float32: return VK_FORMAT_R32_SFLOAT;
    case DataFormat::Rgba32:  return VK_FORMAT_R8G8B8A8_SRGB;
    default:                  return VK_FORMAT_UNDEFINED;
  }
}

VulkanCudaEngine::VulkanCudaEngine(int2 extent, cudaStream_t cuda_stream):
  VulkanEngine(),
  iteration_count(0),
  iteration_idx(0),
  data_extent(extent),
  stream(cuda_stream),
  //cuda_timeline_semaphore(nullptr),
  cuda_wait_semaphore(nullptr),
  cuda_signal_semaphore(nullptr)
{}

VulkanCudaEngine::~VulkanCudaEngine()
{
  if (rendering_thread.joinable())
  {
    rendering_thread.join();
  }
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

  if (texture_sampler != VK_NULL_HANDLE)
  {
    vkDestroySampler(device, texture_sampler, nullptr);
  }

  for (auto& buffer : unstructured_buffers)
  {
    if (buffer.cuda_ptr != nullptr)
    {
      checkCuda(cudaDestroyExternalMemory(buffer.cuda_extmem));
    }
    if (buffer.vk_buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(device, buffer.vk_buffer, nullptr);
    }
    if (buffer.vk_memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, buffer.vk_memory, nullptr);
    }
  }

  for (auto& buffer : structured_buffers)
  {
    if (buffer.cuda_ptr != nullptr)
    {
      checkCuda(cudaDestroyExternalMemory(buffer.cuda_extmem));
    }
    if (buffer.vk_buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(device, buffer.vk_buffer, nullptr);
    }
    if (buffer.vk_memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(device, buffer.vk_memory, nullptr);
    }
    if (buffer.vk_view != VK_NULL_HANDLE)
    {
      vkDestroyImageView(device, buffer.vk_view, nullptr);
    }
    if (buffer.vk_image != VK_NULL_HANDLE)
    {
      vkDestroyImage(device, buffer.vk_image, nullptr);
    }
  }
}

void VulkanCudaEngine::displayAsync()
{
  rendering_thread = std::thread(&VulkanCudaEngine::mainLoopThreaded, this);
}

void VulkanCudaEngine::mainLoopThreaded()
{
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents(); // TODO: Move to main thread

    drawGui();

    std::unique_lock<std::mutex> lock(mutex);
    cond.wait(lock, [&]{ return device_working == false; });

    drawFrame();

    lock.unlock();
  }
  vkDeviceWaitIdle(device);
}

void VulkanCudaEngine::display()
{
  device_working = true;
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    drawGui();

    drawFrame();

    cudaSemaphoreWait();
    if (iteration_idx < iteration_count)
    {
      // Advance the simulation
      step_function();
      iteration_idx++;
    }
    cudaSemaphoreSignal();
  }
  device_working = false;
  vkDeviceWaitIdle(device);
}

void VulkanCudaEngine::registerUnstructuredMemory(void **ptr_devmem,
  size_t elem_count, size_t elem_size)
{
  MappedUnstructuredMemory mapped{};
  mapped.element_count = elem_count;
  mapped.element_size  = elem_size;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = VK_FORMAT_UNDEFINED; //getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;

  // Init unstructured memory
  createExternalBuffer(mapped.element_size * mapped.element_count,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    mapped.vk_buffer, mapped.vk_memory
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.vk_memory, mapped.element_size * mapped.element_count
  );

  unstructured_buffers.push_back(mapped);
  updateDescriptors();
  toggleRenderingMode("unstructured");
  *ptr_devmem = mapped.cuda_ptr;
}

void VulkanCudaEngine::registerStructuredMemory(void **ptr_devmem,
  size_t width, size_t height, size_t elem_size, DataFormat format)
{
  MappedStructuredMemory mapped{};
  mapped.element_count = width * height;
  mapped.element_size  = elem_size;
  mapped.cuda_ptr      = nullptr;
  mapped.cuda_extmem   = nullptr;
  mapped.vk_format     = getVulkanFormat(format);
  mapped.vk_buffer     = VK_NULL_HANDLE;
  mapped.vk_memory     = VK_NULL_HANDLE;
  mapped.vk_image      = VK_NULL_HANDLE;
  mapped.vk_view       = VK_NULL_HANDLE;
  mapped.extent        = { width, height };

  // Init structured memory
  createExternalImage(width, height, mapped.vk_format,
    VK_IMAGE_TILING_LINEAR,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    mapped.vk_image, mapped.vk_memory
  );
  importCudaExternalMemory(&mapped.cuda_ptr, mapped.cuda_extmem,
    mapped.vk_memory, mapped.element_size * width * height
  );

  transitionImageLayout(mapped.vk_image, mapped.vk_format,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );
  mapped.vk_view = createImageView(mapped.vk_image, mapped.vk_format);

  structured_buffers.push_back(mapped);
  updateDescriptors();
  toggleRenderingMode("structured");
  *ptr_devmem = mapped.cuda_ptr;
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
  // Vulkan signal will be CUDA wait
  importCudaExternalSemaphore(cuda_signal_semaphore, vk_wait_semaphore);

  createExternalSemaphore(vk_signal_semaphore);
  // CUDA signal will be vulkan wait
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

void VulkanCudaEngine::cudaSemaphoreWait()
{
  cudaExternalSemaphoreWaitParams wait_params{};
  wait_params.flags = 0;
  wait_params.params.fence.value = 0;
  // Wait for Vulkan to complete its work
  checkCuda(cudaWaitExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &cuda_wait_semaphore, &wait_params, 1, stream)
  );
}

void VulkanCudaEngine::cudaSemaphoreSignal()
{
  cudaExternalSemaphoreSignalParams signal_params{};
  signal_params.flags = 0;
  signal_params.params.fence.value = 0;

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
  if (current_frame != 0 && device_working == true)
  {
    // Vulkan waits until Cuda is done with the display buffer before rendering
    wait.push_back(vk_wait_semaphore);
    // Cuda will wait until all pipeline commands are complete
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
}

void VulkanCudaEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
  // Vulkan will signal to this semaphore once the device array is ready
  // for Cuda to process
  signal.push_back(vk_signal_semaphore);
}

void VulkanCudaEngine::setUnstructuredRendering(VkCommandBuffer& cmd_buffer)
{
  VkBuffer vertex_buffers[] = { unstructured_buffers[0].vk_buffer };
  VkDeviceSize offsets[] = { 0 };
  auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
  vkCmdBindVertexBuffers(cmd_buffer, 0, binding_count, vertex_buffers, offsets);
  vkCmdDraw(cmd_buffer, unstructured_buffers[0].element_count, 1, 0, 0);
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

void VulkanCudaEngine::updateUniformBuffer(uint32_t image_index)
{
  VulkanEngine::updateUniformBuffer(image_index);

  SceneParams params{};
  params.extent = glm::ivec2{data_extent.x, data_extent.y};

  void *data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], sizeof(ModelViewProjection),
    sizeof(SceneParams), 0, &data
  );
  memcpy(data, &params, sizeof(params));
  vkUnmapMemory(device, ubo_memory[image_index]);
}

void VulkanCudaEngine::prepareWindow()
{
  std::unique_lock<std::mutex> ul(mutex);
  device_working = true;
  cudaSemaphoreWait();
  ul.unlock();
  cond.notify_one();
}

void VulkanCudaEngine::updateWindow()
{
  std::unique_lock<std::mutex> ul(mutex);
  device_working = false;
  cudaSemaphoreSignal();
  ul.unlock();
  cond.notify_one();
}

void VulkanCudaEngine::updateDescriptors()
{
  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    // Write MVP matrix, scene info and texture samplers
    std::vector<VkWriteDescriptorSet> desc_writes;
    desc_writes.reserve(2 + structured_buffers.size());

    VkDescriptorBufferInfo mvp_info{};
    mvp_info.buffer = uniform_buffers[i];
    mvp_info.offset = 0;
    mvp_info.range  = sizeof(ModelViewProjection);

    VkWriteDescriptorSet write_mvp{};
    write_mvp.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_mvp.dstSet           = descriptor_sets[i];
    write_mvp.dstBinding       = 0;
    write_mvp.dstArrayElement  = 0;
    write_mvp.descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write_mvp.descriptorCount  = 1;
    write_mvp.pBufferInfo      = &mvp_info;
    write_mvp.pImageInfo       = nullptr;
    write_mvp.pTexelBufferView = nullptr;
    desc_writes.push_back(write_mvp);

    VkDescriptorBufferInfo extent_info{};
    extent_info.buffer = uniform_buffers[i];
    extent_info.offset = sizeof(ModelViewProjection);
    extent_info.range  = sizeof(SceneParams);

    VkWriteDescriptorSet write_scene{};
    write_scene.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_scene.dstSet           = descriptor_sets[i];
    write_scene.dstBinding       = 1;
    write_scene.dstArrayElement  = 0;
    write_scene.descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write_scene.descriptorCount  = 1;
    write_scene.pBufferInfo      = &extent_info;
    write_scene.pImageInfo       = nullptr;
    write_scene.pTexelBufferView = nullptr;
    desc_writes.push_back(write_scene);

    for (const auto& buffer : structured_buffers)
    {
      VkDescriptorImageInfo image_info{};
      image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      image_info.imageView   = buffer.vk_view;
      image_info.sampler     = texture_sampler;

      VkWriteDescriptorSet write_tex{};
      write_tex.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_tex.dstSet          = descriptor_sets[i];
      write_tex.dstBinding      = 2;
      write_tex.dstArrayElement = 0;
      write_tex.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      write_tex.descriptorCount = 1;
      write_tex.pImageInfo      = &image_info;
      desc_writes.push_back(write_tex);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}
