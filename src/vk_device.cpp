#include "internal/vk_device.hpp"

#include <set>

#include "internal/validation.hpp"
#include "internal/vk_properties.hpp"
#include "internal/vk_initializers.hpp"

VulkanDevice::VulkanDevice(VkPhysicalDevice gpu): physical_device{gpu}
{
  vkGetPhysicalDeviceProperties(physical_device, &properties);
  vkGetPhysicalDeviceFeatures(physical_device, &features);
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
}

VulkanDevice::~VulkanDevice()
{
  deletors.flush();
}

void VulkanDevice::initLogicalDevice(VkSurfaceKHR surface)
{
  props::findQueueFamilies(
    physical_device, surface, queue_indices.graphics, queue_indices.present
  );

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set unique_queue_families{ queue_indices.graphics, queue_indices.present };
  auto queue_priority = 1.f;

  for (auto queue_family : unique_queue_families)
  {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount       = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  VkPhysicalDeviceFeatures device_features{};
  device_features.samplerAnisotropy = VK_TRUE;
  device_features.fillModeNonSolid  = VK_TRUE; // Enable wireframe

  // Explicitly enable timeline semaphores, or validation layer will complain
  VkPhysicalDeviceVulkan12Features features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features.timelineSemaphore = true;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pNext = &features;
  create_info.queueCreateInfoCount = queue_create_infos.size();
  create_info.pQueueCreateInfos    = queue_create_infos.data();
  create_info.pEnabledFeatures     = &device_features;

  auto device_extensions = props::getRequiredDeviceExtensions();
  create_info.enabledExtensionCount   = device_extensions.size();
  create_info.ppEnabledExtensionNames = device_extensions.data();

  if (validation::enable_layers)
  {
    create_info.enabledLayerCount   = validation::layers.size();
    create_info.ppEnabledLayerNames = validation::layers.data();
  }
  else
  {
    create_info.enabledLayerCount = 0;
  }

  validation::checkVulkan(vkCreateDevice(
    physical_device, &create_info, nullptr, &logical_device)
  );
  deletors.pushFunction([=](){
    vkDestroyDevice(logical_device, nullptr);
  });

  vkGetDeviceQueue(logical_device, queue_indices.graphics, 0, &queues.graphics);
  vkGetDeviceQueue(logical_device, queue_indices.present, 0, &queues.present);

  // TODO: Get device UUID for cuda

  command_pool = createCommandPool(queue_indices.graphics,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
  );
}

VkCommandPool VulkanDevice::createCommandPool(
  uint32_t queue_idx, VkCommandPoolCreateFlags flags)
{
  VkCommandPool new_pool = VK_NULL_HANDLE;
  auto pool_info = vkinit::commandPoolCreateInfo(queue_idx, flags);
  validation::checkVulkan(vkCreateCommandPool(
    logical_device, &pool_info, nullptr, &new_pool)
  );
  deletors.pushFunction([=](){
    vkDestroyCommandPool(logical_device, command_pool, nullptr);
  });

  return new_pool;
}

std::vector<VkCommandBuffer> VulkanDevice::createCommandBuffers(uint32_t buffer_count)
{
  std::vector<VkCommandBuffer> buffers(buffer_count, VK_NULL_HANDLE);
  auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool, buffer_count);
  validation::checkVulkan(vkAllocateCommandBuffers(
    logical_device, &alloc_info, buffers.data())
  );
  return buffers;
}

VkCommandBuffer VulkanDevice::beginSingleTimeCommands()
{
  VkCommandBuffer cmd;
  auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool);
  validation::checkVulkan(vkAllocateCommandBuffers(logical_device, &alloc_info, &cmd));

  // Begin command buffer recording with a only-one-use buffer
  auto flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  auto begin_info = vkinit::commandBufferBeginInfo(flags);

  validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));
  return cmd;
}

void VulkanDevice::endSingleTimeCommands(VkCommandBuffer cmd, VkQueue queue)
{
  // Finish recording the command buffer
  validation::checkVulkan(vkEndCommandBuffer(cmd));

  auto submit_info = vkinit::submitInfo(&cmd);
  validation::checkVulkan(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(logical_device, command_pool, 1, &cmd);
}

void VulkanDevice::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  validation::checkVulkan(
    vkCreateBuffer(logical_device, &buffer_info, nullptr, &buffer)
  );

  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(logical_device, buffer, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  auto type = findMemoryType(mem_req.memoryTypeBits, properties);
  alloc_info.memoryTypeIndex = type;
  validation::checkVulkan(
    vkAllocateMemory(logical_device, &alloc_info, nullptr, &memory)
  );

  vkBindBufferMemory(logical_device, buffer, memory, 0);
}

void VulkanDevice::createExternalBuffer(VkDeviceSize size,
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

  validation::checkVulkan(
    vkCreateBuffer(logical_device, &buffer_info, nullptr, &buffer)
  );

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(logical_device, buffer, &mem_reqs);
  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_info;
  alloc_info.allocationSize = mem_reqs.size;
  auto type = findMemoryType(mem_reqs.memoryTypeBits, properties);
  alloc_info.memoryTypeIndex = type;

  validation::checkVulkan(vkAllocateMemory(
    logical_device, &alloc_info, nullptr, &buffer_memory)
  );
  vkBindBufferMemory(logical_device, buffer, buffer_memory, 0);
}

void VulkanDevice::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size, VkQueue queue)
{
  // Memory transfers are commands executed with buffers, just like drawing
  auto cmd_buffer = beginSingleTimeCommands();

  VkBufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;
  vkCmdCopyBuffer(cmd_buffer, src, dst, 1, &copy_region);

  endSingleTimeCommands(cmd_buffer, queue);
}

uint32_t VulkanDevice::findMemoryType(
  uint32_t type_filter, VkMemoryPropertyFlags properties)
{
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
  {
    auto flags = memory_properties.memoryTypes[i].propertyFlags;
    if ((type_filter & (1 << i)) && (flags & properties) == properties)
    {
      return i;
    }
  }
  return ~0;
}
