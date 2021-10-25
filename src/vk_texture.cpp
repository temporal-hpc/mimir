#include "cudaview/vk_engine.hpp"
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

  VkImageCreateInfo image_info{};
  image_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.pNext         = &ext_info;
  image_info.imageType     = VK_IMAGE_TYPE_2D;
  image_info.extent.width  = width;
  image_info.extent.height = height;
  image_info.extent.depth  = 1;
  image_info.mipLevels     = 1;
  image_info.arrayLayers   = 1;
  image_info.format        = format;
  image_info.tiling        = tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage         = usage;
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

void VulkanEngine::createImage(uint32_t width, uint32_t height, VkFormat format,
  VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
  VkImage& image, VkDeviceMemory& image_memory)
{
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType     = VK_IMAGE_TYPE_2D;
  image_info.extent.width  = width;
  image_info.extent.height = height;
  image_info.extent.depth  = 1;
  image_info.mipLevels     = 1;
  image_info.arrayLayers   = 1;
  image_info.format        = format;
  image_info.tiling        = tiling;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage         = usage;
  image_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
  image_info.flags         = 0;

  validation::checkVulkan(vkCreateImage(device, &image_info, nullptr, &image));
  VkMemoryRequirements mem_req;
  vkGetImageMemoryRequirements(device, image, &mem_req);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex =
    utils::findMemoryType(physical_device, mem_req.memoryTypeBits, properties);

  validation::checkVulkan(
    vkAllocateMemory(device, &alloc_info, nullptr, &image_memory)
  );
  vkBindImageMemory(device, image, image_memory, 0);
}

void VulkanEngine::createTextureImage()
{
  std::string filename = "textures/lena512.png";
  int tex_width, tex_height, tex_channels;
  auto pixels = io::loadImage(filename, tex_width, tex_height, tex_channels);
  if (!pixels)
  {
    throw std::runtime_error("failed to load texture image");
  }
  VkDeviceSize img_size = tex_width * tex_height * 4;

  createBuffer(img_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    staging_buffer, staging_memory
  );

  void *data;
  vkMapMemory(device, staging_memory, 0, img_size, 0, &data);
  memcpy(data, pixels, static_cast<size_t>(img_size));
  vkUnmapMemory(device, staging_memory);

  stbi_image_free(pixels);

  createImage(tex_width, tex_height,
    VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    texture_image, texture_memory
  );

  transitionImageLayout(texture_image, VK_FORMAT_R8G8B8A8_SRGB,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
  );
  copyBufferToImage(staging_buffer, texture_image, tex_width, tex_height);
  transitionImageLayout(texture_image, VK_FORMAT_R8G8B8A8_SRGB,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
  );

  vkDestroyBuffer(device, staging_buffer, nullptr);
  vkFreeMemory(device, staging_memory, nullptr);
}

void VulkanEngine::transitionImageLayout(VkImage image, VkFormat format,
  VkImageLayout old_layout, VkImageLayout new_layout)
{
  auto cmd_buffer = beginSingleTimeCommands();

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
  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
           new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else
  {
    throw std::invalid_argument("unsupported layout transition");
  }

  vkCmdPipelineBarrier(cmd_buffer, src_stage, dst_stage,
    0, 0, nullptr, 0, nullptr, 1, &barrier
  );

  endSingleTimeCommands(cmd_buffer);
}

void VulkanEngine::copyBufferToImage(VkBuffer buffer, VkImage image,
  uint32_t width, uint32_t height)
{
  auto cmd_buffer = beginSingleTimeCommands();

  VkBufferImageCopy region{};
  region.bufferOffset      = 0;
  region.bufferRowLength   = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel       = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount     = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(cmd_buffer, buffer, image,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region
  );

  endSingleTimeCommands(cmd_buffer);
}

VkImageView VulkanEngine::createImageView(VkImage image, VkFormat format)
{
  VkImageViewCreateInfo view_info{};
  view_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image    = image;
  // Treat image as 1D/2D/3D texture or as a cube map
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format   = format;
  // Default mapping of all color channels
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  // Describe image purpose and which part of it should be accesssed
  view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel   = 0;
  view_info.subresourceRange.levelCount     = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount     = 1;

  VkImageView image_view;
  validation::checkVulkan(
    vkCreateImageView(device, &view_info, nullptr, &image_view)
  );
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

void VulkanEngine::createTextureImageView()
{
  texture_view = createImageView(texture_image, VK_FORMAT_R8G8B8A8_SRGB);
}

void VulkanEngine::createTextureSampler()
{
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device, &properties);

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.anisotropyEnable = VK_TRUE;
  sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
  sampler_info.borderColor   = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp  = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.mipLodBias = 0.f;
  sampler_info.minLod = 0.f;
  sampler_info.maxLod = 0.f;

  validation::checkVulkan(
    vkCreateSampler(device, &sampler_info, nullptr, &texture_sampler)
  );
}
