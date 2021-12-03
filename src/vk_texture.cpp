#include "cudaview/vk_engine.hpp"
#include "vk_initializers.hpp"
#include "validation.hpp"
#include "io.hpp"
#include "utils.hpp"

#include <cstring> // memcpy

void VulkanEngine::createExternalImage(uint32_t width, uint32_t height,
  VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
  VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& image_memory)
{
  VkExternalMemoryImageCreateInfo ext_info{};
  ext_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  ext_info.pNext = nullptr;
  ext_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  auto image_info = vkinit::imageCreateInfo(format, usage, width, height);
  image_info.pNext         = &ext_info;
  image_info.format        = format;
  image_info.tiling        = tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
  image_info.flags         = 0;

  validation::checkVulkan(vkCreateImage(device, &image_info, nullptr, &image));

  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(device, image, &mem_req);

  VkExportMemoryAllocateInfoKHR export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
  export_info.pNext = nullptr;
  export_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &export_info;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex =
    utils::findMemoryType(physical_device, mem_req.memoryTypeBits, properties);

  validation::checkVulkan(
    vkAllocateMemory(device, &alloc_info, nullptr, &image_memory)
  );
  vkBindImageMemory(device, image, image_memory, 0);
}

void VulkanEngine::transitionImageLayout(VkImage image, VkFormat format,
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

  auto cmd_buffer = beginSingleTimeCommands();
  vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage,
    0, 0, nullptr, 0, nullptr, 1, &barrier
  );
  endSingleTimeCommands(cmd_buffer);
}

VkImageView VulkanEngine::createImageView(VkImage image, VkFormat format)
{
  auto info = vkinit::imageViewCreateInfo(format, image, VK_IMAGE_ASPECT_COLOR_BIT);
  VkImageView image_view;
  validation::checkVulkan(vkCreateImageView(device, &info, nullptr, &image_view));
  return image_view;
}

// Set up image views, so they can be used as color targets later on
void VulkanEngine::createImageViews()
{
  swapchain_views.resize(swapchain_images.size());
  // Create a basic image view for every image in the swap chain
  for (size_t i = 0; i < swapchain_images.size(); ++i)
  {
    swapchain_views[i] = createImageView(swapchain_images[i], swapchain_format);
  }
}

void VulkanEngine::createTextureSampler()
{
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);

  auto sampler_info = vkinit::samplerCreateInfo(VK_FILTER_LINEAR);
  sampler_info.anisotropyEnable        = VK_TRUE;
  sampler_info.maxAnisotropy           = properties.limits.maxSamplerAnisotropy;
  sampler_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable           = VK_FALSE;
  sampler_info.compareOp               = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.mipLodBias              = 0.f;
  sampler_info.minLod                  = 0.f;
  sampler_info.maxLod                  = 0.f;

  validation::checkVulkan(
    vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler)
  );
}
