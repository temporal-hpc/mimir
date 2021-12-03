#include "cudaview/vk_engine.hpp"
#include "vk_initializers.hpp"
#include "validation.hpp"

void VulkanEngine::createCommandPool()
{
  uint32_t graphics_index, present_index;
  findQueueFamilies(physical_device, graphics_index, present_index);
  auto pool_info = vkinit::commandPoolCreateInfo(
    graphics_index, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  validation::checkVulkan(vkCreateCommandPool(
    device, &pool_info, nullptr, &command_pool)
  );
}

void VulkanEngine::createCommandBuffers()
{
  command_buffers.resize(framebuffers.size());

  auto buffer_count = static_cast<uint32_t>(command_buffers.size());
  auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool, buffer_count);

  validation::checkVulkan(vkAllocateCommandBuffers(
    device, &alloc_info, command_buffers.data())
  );
}

VkCommandBuffer VulkanEngine::beginSingleTimeCommands()
{
  VkCommandBuffer cmd;
  auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool);
  validation::checkVulkan(vkAllocateCommandBuffers(device, &alloc_info, &cmd));

  // Begin command buffer recording with a only-one-use buffer
  auto flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  auto begin_info = vkinit::commandBufferBeginInfo(flags);

  validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));
  return cmd;
}

void VulkanEngine::endSingleTimeCommands(VkCommandBuffer cmd)
{
  // Finish recording the command buffer
  validation::checkVulkan(vkEndCommandBuffer(cmd));

  auto submit_info = vkinit::submitInfo(&cmd);
  validation::checkVulkan(vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE));
  vkQueueWaitIdle(graphics_queue);
  vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}
