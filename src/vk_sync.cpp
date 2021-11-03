#include "cudaview/vk_engine.hpp"
#include "validation.hpp"

static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

void VulkanEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
  std::vector<VkPipelineStageFlags>& wait_stages) const
{}

void VulkanEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{}


void VulkanEngine::createSyncObjects() // TODO: add inflight_frame_count as parameter
{
  images_inflight.resize(swapchain_images.size(), VK_NULL_HANDLE);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  /*validation::checkVulkan(vkCreateSemaphore(
    device, &semaphore_info, nullptr, &vk_presentation_semaphore)
  );*/
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
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

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
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
  fd_info.semaphore = semaphore;
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
