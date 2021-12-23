#include "cudaview/vk_engine.hpp"
#include "validation.hpp"
#include "vk_properties.hpp"

#include "cudaview/vk_types.hpp"
#include "cudaview/camera.hpp"

#include "glm/gtc/type_ptr.hpp"

#include <cstring> // memcpy

void VulkanEngine::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory &memory)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  validation::checkVulkan(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(device, buffer, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  auto type = props::findMemoryType(physical_device, mem_req.memoryTypeBits, properties);
  alloc_info.memoryTypeIndex = type;
  validation::checkVulkan(vkAllocateMemory(device, &alloc_info, nullptr, &memory));

  vkBindBufferMemory(device, buffer, memory, 0);
}

void VulkanEngine::createExternalBuffer(VkDeviceSize size,
  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
  VkExternalMemoryHandleTypeFlagsKHR handle_type, VkBuffer& buffer,
  VkDeviceMemory& buffer_memory)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo extmem_info{};
  extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  extmem_info.handleTypes = handle_type;
  buffer_info.pNext = &extmem_info;

  validation::checkVulkan(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);
  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_info;
  alloc_info.allocationSize = mem_reqs.size;
  auto type = props::findMemoryType(physical_device, mem_reqs.memoryTypeBits, properties);
  alloc_info.memoryTypeIndex = type;

  validation::checkVulkan(vkAllocateMemory(
    device, &alloc_info, nullptr, &buffer_memory)
  );
  vkBindBufferMemory(device, buffer, buffer_memory, 0);
}

void VulkanEngine::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
  // Memory transfers are commands executed with buffers, just like drawing
  auto cmd_buffer = beginSingleTimeCommands();

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;
  vkCmdCopyBuffer(cmd_buffer, src, dst, 1, &copy_region);

  endSingleTimeCommands(cmd_buffer);
}

void *VulkanEngine::getMemoryHandle(VkDeviceMemory memory,
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

void VulkanEngine::createUniformBuffers()
{
  VkDeviceSize buffer_size = sizeof(ModelViewProjection) + 2 * 0x40;
  auto img_count = swapchain_images.size();
  uniform_buffers.resize(img_count);
  ubo_memory.resize(img_count);

  // TODO: Use a single buffer of "img_count * buffer_size" bytes
  for (size_t i = 0; i < img_count; ++i)
  {
    createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      uniform_buffers[i], ubo_memory[i]
    );
  }
}

void VulkanEngine::updateUniformBuffer(uint32_t image_index)
{
  ModelViewProjection ubo{};
  ubo.model = glm::mat4(1.f);
  ubo.view  = camera.matrices.view; // glm::mat4(1.f);
  ubo.proj  = camera.matrices.perspective; //glm::mat4(1.f);

  void *data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], 0, sizeof(ubo), 0, &data);
  memcpy(data, &ubo, sizeof(ubo));
  vkUnmapMemory(device, ubo_memory[image_index]);

  ColorParams colors{};
  colors.point_color = setColor(point_color);
  colors.edge_color  = setColor(edge_color);
  data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], sizeof(ubo),
    sizeof(ColorParams), 0, &data
  );
  memcpy(data, &colors, sizeof(colors));
  vkUnmapMemory(device, ubo_memory[image_index]);

  SceneParams params{};
  params.extent = glm::ivec3{data_extent.x, data_extent.y, data_extent.z};
  // TODO: Merge mappings
  data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], sizeof(ubo) + 0x40,
    sizeof(SceneParams), 0, &data
  );
  memcpy(data, &params, sizeof(params));
  vkUnmapMemory(device, ubo_memory[image_index]);
}
