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
  auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool);
  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device, &alloc_info, &command_buffer);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer, &begin_info);
  return command_buffer;
}

void VulkanEngine::endSingleTimeCommands(VkCommandBuffer command_buffer)
{
  // Finish recording the command buffer
  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers    = &command_buffer;

  vkQueueSubmit(graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue);
  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}
