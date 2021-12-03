#include "cudaview/vk_engine.hpp"
#include "vk_initializers.hpp"
#include "validation.hpp"

static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

void VulkanEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
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

void VulkanEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
  // Vulkan will signal to this semaphore once the device array is ready
  // for Cuda to process
  signal.push_back(vk_signal_semaphore);
}

void VulkanEngine::createSyncObjects() // TODO: add inflight_frame_count as param
{
  images_inflight.resize(swapchain_images.size(), VK_NULL_HANDLE);

  /*validation::checkVulkan(vkCreateSemaphore(
    device, &semaphore_info, nullptr, &vk_presentation_semaphore)
  );*/
  auto semaphore_info = vkinit::semaphoreCreateInfo();
  auto fence_info = vkinit::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
  {
    validation::checkVulkan(vkCreateSemaphore(
      device, &semaphore_info, nullptr, &image_available[i])
    );
    validation::checkVulkan(vkCreateSemaphore(
      device, &semaphore_info, nullptr, &render_finished[i])
    );
    validation::checkVulkan(vkCreateFence(
      device, &fence_info, nullptr, &inflight_fences[i])
    );
  }

  //createExternalSemaphore(vk_timeline_semaphore);
  //importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);

  createExternalSemaphore(vk_wait_semaphore);
  // Vulkan signal will be CUDA wait
  importCudaExternalSemaphore(cuda_signal_semaphore, vk_wait_semaphore);

  createExternalSemaphore(vk_signal_semaphore);
  // CUDA signal will be vulkan wait
  importCudaExternalSemaphore(cuda_wait_semaphore, vk_signal_semaphore);
}

void VulkanEngine::createExternalSemaphore(VkSemaphore& semaphore)
{
  /*VkSemaphoreTypeCreateInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_info.pNext = nullptr;
  timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_info.initialValue = 0;*/

  VkExportSemaphoreCreateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  export_info.pNext = nullptr; // &timeline_info

  auto semaphore_info = vkinit::semaphoreCreateInfo();
  semaphore_info.pNext = &export_info;

  validation::checkVulkan(
    vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore)
  );
}

void *VulkanEngine::getSemaphoreHandle(VkSemaphore semaphore,
  VkExternalSemaphoreHandleTypeFlagBits handle_type)
{
  int fd;
  VkSemaphoreGetFdInfoKHR fd_info{};
  fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  fd_info.pNext = nullptr;
  fd_info.semaphore  = semaphore;
  fd_info.handleType = handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphore;
  fpGetSemaphore = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
    device, "vkGetSemaphoreFdKHR"
  );
  if (!fpGetSemaphore)
  {
    throw std::runtime_error("Failed to retrieve semaphore function handle!");
  }
  validation::checkVulkan(fpGetSemaphore(device, &fd_info, &fd));

  return (void*)(uintptr_t)fd;
}
