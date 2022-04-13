#include "cudaview/engine/vk_device.hpp"

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
    physical_device, surface, graphics.family_index, present.family_index
  );

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set unique_queue_families{ graphics.family_index, present.family_index };
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

  vkGetDeviceQueue(logical_device, graphics.family_index, 0, &graphics.queue);
  vkGetDeviceQueue(logical_device, present.family_index, 0, &present.queue);

  // TODO: Get device UUID for cuda

  command_pool = createCommandPool(graphics.family_index,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
  );
}

VkCommandPool VulkanDevice::createCommandPool(
  uint32_t queue_idx, VkCommandPoolCreateFlags flags)
{
  VkCommandPool new_pool = VK_NULL_HANDLE;
  auto pool_info = vkinit::commandPoolCreateInfo(flags, queue_idx);
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
  auto alloc_info = vkinit::commandBufferAllocateInfo(
    command_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, buffer_count
  );
  validation::checkVulkan(vkAllocateCommandBuffers(
    logical_device, &alloc_info, buffers.data())
  );
  return buffers;
}

void VulkanDevice::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
  auto queue = graphics.queue;
  VkCommandBuffer cmd;
  auto alloc_info = vkinit::commandBufferAllocateInfo(command_pool);
  validation::checkVulkan(vkAllocateCommandBuffers(logical_device, &alloc_info, &cmd));

  // Begin command buffer recording with a only-one-use buffer
  auto flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  auto begin_info = vkinit::commandBufferBeginInfo(flags);

  validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));
  function(cmd);
  validation::checkVulkan(vkEndCommandBuffer(cmd));

  auto submit_info = vkinit::submitInfo(&cmd);
  validation::checkVulkan(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(logical_device, command_pool, 1, &cmd);
}

VulkanBuffer VulkanDevice::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags properties)
{
  return createBuffer(size, usage, properties, nullptr, nullptr);
}

VulkanBuffer VulkanDevice::createExternalBuffer(VkDeviceSize size,
  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
  VkExternalMemoryHandleTypeFlagsKHR handle_type)
{
  VkExternalMemoryBufferCreateInfo extmem_info{};
  extmem_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  extmem_info.handleTypes = handle_type;

  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  return createBuffer(size, usage, properties, &extmem_info, &export_info);
}

VulkanTexture VulkanDevice::createExternalImage(VkImageType type,
  VkFormat format, VkExtent3D extent, VkImageTiling tiling,
  VkImageUsageFlags usage, VkMemoryPropertyFlags mem_props)
{
  VkExternalMemoryImageCreateInfo ext_info{};
  ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  ext_info.pNext = nullptr;
  ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  // TODO: Check if texture is within bounds
  //auto max_img_dim = properties.limits.maxImageDimension3D;

  auto image_info = vkinit::imageCreateInfo(type, format, extent, usage);
  image_info.pNext         = &ext_info;
  image_info.flags         = 0;
  image_info.tiling        = tiling;
  image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VulkanTexture tex;
  tex.width  = extent.width;
  tex.height = extent.height;
  tex.depth  = extent.depth;
  validation::checkVulkan(
    vkCreateImage(logical_device, &image_info, nullptr, &tex.image)
  );

  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(logical_device, tex.image, &mem_req);

  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_info;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = findMemoryType(mem_req.memoryTypeBits, mem_props);

  validation::checkVulkan(
    vkAllocateMemory(logical_device, &alloc_info, nullptr, &tex.memory)
  );
  vkBindImageMemory(logical_device, tex.image, tex.memory, 0);

  return tex;
}

VkSemaphore VulkanDevice::createExternalSemaphore()
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

  VkSemaphore semaphore;
  validation::checkVulkan(
    vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &semaphore)
  );
  return semaphore;
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

VulkanBuffer VulkanDevice::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
  VkMemoryPropertyFlags mem_props, const void *extmem_info, const void *export_info)
{
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size  = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_info.pNext = extmem_info;

  VulkanBuffer new_buffer;
  new_buffer.size = size;
  validation::checkVulkan(
    vkCreateBuffer(logical_device, &buffer_info, nullptr, &new_buffer.buffer)
  );

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(logical_device, new_buffer.buffer, &mem_reqs);
  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = export_info;
  alloc_info.allocationSize = mem_reqs.size;
  auto type = findMemoryType(mem_reqs.memoryTypeBits, mem_props);
  alloc_info.memoryTypeIndex = type;

  validation::checkVulkan(vkAllocateMemory(
    logical_device, &alloc_info, nullptr, &new_buffer.memory)
  );
  vkBindBufferMemory(logical_device, new_buffer.buffer, new_buffer.memory, 0);
  return new_buffer;
}

void VulkanDevice::transitionImageLayout(VkImage image, VkFormat format,
  VkImageLayout old_layout, VkImageLayout new_layout)
{
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = 0;

  VkPipelineStageFlags src_stage, dst_stage;
  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED)
  {
    barrier.srcAccessMask = 0;
    src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  }
  else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else
  {
    throw std::invalid_argument("unsupported layout transition");
  }

  if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  {
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else
  {
    throw std::invalid_argument("unsupported layout transition");
  }

  immediateSubmit([=](VkCommandBuffer cmd)
  {
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  });
}
