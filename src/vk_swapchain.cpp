#include "cudaview/vk_engine.hpp"
#include "vk_initializers.hpp"
#include "vk_pipeline.hpp"
#include "vk_properties.hpp"
#include "validation.hpp"
#include "io.hpp"

#include <algorithm> // std::clamp
#include <filesystem> // std::filesystem

#include "cudaview/vk_types.hpp"

VkSurfaceFormatKHR chooseSwapSurfaceFormat(
  const std::vector<VkSurfaceFormatKHR>& available_formats)
{
  for (const auto& available_format : available_formats)
  {
    if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      return available_format;
    }
  }
  return available_formats[0];
}

VkPresentModeKHR chooseSwapPresentMode(
  const std::vector<VkPresentModeKHR>& available_modes)
{
  VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (const auto& available_mode : available_modes)
  {
    if (available_mode == VK_PRESENT_MODE_MAILBOX_KHR)
    {
      return available_mode;
    }
    else if (available_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
    {
      best_mode = available_mode;
    }
  }
  return best_mode;
}

void VulkanEngine::cleanupSwapchain()
{
  for (size_t i = 0; i < swapchain_images.size(); ++i)
  {
    vkDestroyBuffer(device, uniform_buffers[i], nullptr);
    vkFreeMemory(device, ubo_memory[i], nullptr);
  }
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkFreeCommandBuffers(device, command_pool,
    static_cast<uint32_t>(command_buffers.size()), command_buffers.data()
  );
  vkDestroyPipeline(device, graphics_pipeline, nullptr);
  vkDestroyPipeline(device, screen_pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  for (auto framebuffer : framebuffers)
  {
    vkDestroyFramebuffer(device, framebuffer, nullptr);
  }
  vkDestroyRenderPass(device, render_pass, nullptr);
  for (auto image_view : swapchain_views)
  {
    vkDestroyImageView(device, image_view, nullptr);
  }
  vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VulkanEngine::initSwapchain()
{
  createSwapChain();
  createImageViews();
  createRenderPass();
  createGraphicsPipelines();
  createFramebuffers();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  updateDescriptors();
}

void VulkanEngine::recreateSwapchain()
{
  vkDeviceWaitIdle(device);

  cleanupSwapchain();
  initSwapchain();
}

VkExtent2D VulkanEngine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
  {
    return capabilities.currentExtent;
  }
  else
  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D actual_extent = {
      static_cast<uint32_t>(width), static_cast<uint32_t>(height)
    };
    actual_extent.width = std::clamp(actual_extent.width,
      capabilities.minImageExtent.width, capabilities.maxImageExtent.width
    );
    actual_extent.height = std::clamp(actual_extent.height,
      capabilities.minImageExtent.height, capabilities.maxImageExtent.height
    );
    return actual_extent;
  }
}

void VulkanEngine::createSwapChain()
{
  auto swapchain_support = props::getSwapchainProperties(physical_device, surface);
  auto surface_format = chooseSwapSurfaceFormat(swapchain_support.formats);
  auto present_mode = chooseSwapPresentMode(swapchain_support.present_modes);
  auto extent = chooseSwapExtent(swapchain_support.capabilities);

  auto image_count = swapchain_support.capabilities.minImageCount + 1;
  const auto max_image_count = swapchain_support.capabilities.maxImageCount;
  if (max_image_count > 0 && image_count > max_image_count)
  {
    image_count = max_image_count;
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface          = surface;
  create_info.minImageCount    = image_count;
  create_info.imageFormat      = surface_format.format;
  create_info.imageColorSpace  = surface_format.colorSpace;
  create_info.imageExtent      = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queue_ids[2];
  props::findQueueFamilies(physical_device, surface, queue_ids[0], queue_ids[1]);

  if (queue_ids[0] != queue_ids[1])
  {
    create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices   = queue_ids;
  }
  else
  {
    create_info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices   = nullptr;
  }
  create_info.preTransform   = swapchain_support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode    = present_mode;
  create_info.clipped        = VK_TRUE;
  create_info.oldSwapchain   = VK_NULL_HANDLE;

  validation::checkVulkan(vkCreateSwapchainKHR(
    device, &create_info, nullptr, &swapchain)
  );

  vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
  swapchain_images.resize(image_count);
  vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());

  swapchain_format = surface_format.format;
  swapchain_extent = extent;
}

void VulkanEngine::createFramebuffers()
{
  framebuffers.resize(swapchain_views.size());
  for (size_t i = 0; i < swapchain_views.size(); ++i)
  {
    VkImageView attachments[] = { swapchain_views[i] };
    auto fb_info = vkinit::framebufferCreateInfo(render_pass, swapchain_extent);
    fb_info.pAttachments    = attachments;

    validation::checkVulkan(vkCreateFramebuffer(
      device, &fb_info, nullptr, &framebuffers[i])
    );
  }
}

void VulkanEngine::createDescriptorPool()
{
  std::array<VkDescriptorPoolSize, 2> pool_sizes{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_sizes[0].descriptorCount = swapchain_images.size();
  pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_sizes[1].descriptorCount = swapchain_images.size();

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes    = pool_sizes.data();
  pool_info.maxSets       = swapchain_images.size();
  pool_info.flags         = 0;

  validation::checkVulkan(
    vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool)
  );
}

// Take buffer with shader bytecode and create a shader module from it
VkShaderModule VulkanEngine::createShaderModule(const std::vector<char>& code)
{
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule module;
  validation::checkVulkan(vkCreateShaderModule(device, &create_info, nullptr, &module));

  return module;
}

void VulkanEngine::createGraphicsPipelines()
{
  // Linux-only hack: Change current directory as quick fix for finding shader paths
  auto orig_path = std::filesystem::current_path();
  auto exec_path = std::filesystem::read_symlink("/proc/self/exe").remove_filename();
  std::filesystem::current_path(exec_path);

  auto vert_code   = io::readFile("shaders/unstructured/particle_pos.spv");
  auto vert_module = createShaderModule(vert_code);

  auto frag_code   = io::readFile("shaders/unstructured/particle_draw.spv");
  auto frag_module = createShaderModule(frag_code);

  auto vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );
  auto frag_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_FRAGMENT_BIT, frag_module
  );

  std::vector<VkDescriptorSetLayout> layouts{descriptor_layout};
  auto pipeline_layout_info = vkinit::pipelineLayoutCreateInfo(layouts);

  validation::checkVulkan(vkCreatePipelineLayout(
    device, &pipeline_layout_info, nullptr, &pipeline_layout)
  );

  std::vector<VkVertexInputBindingDescription> bind_desc;
  std::vector<VkVertexInputAttributeDescription> attr_desc;
  getVertexDescriptions(bind_desc, attr_desc);

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  getAssemblyStateInfo(input_assembly);

  PipelineBuilder builder;
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  builder.input_assembly    = input_assembly;
  builder.viewport.x        = 0.f;
  builder.viewport.y        = 0.f;
  builder.viewport.width    = static_cast<float>(swapchain_extent.width);
  builder.viewport.height   = static_cast<float>(swapchain_extent.height);
  builder.viewport.minDepth = 0.f;
  builder.viewport.maxDepth = 1.f;
  builder.scissor.offset    = {0, 0};
  builder.scissor.extent    = swapchain_extent;
  builder.rasterizer        = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  builder.multisampling     = vkinit::multisamplingStateCreateInfo();
  builder.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.pipeline_layout   = pipeline_layout;
  graphics_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);

  vert_code   = io::readFile("shaders/structured/screen_triangle.spv");
  vert_module = createShaderModule(vert_code);

  frag_code   = io::readFile("shaders/structured/texture_greyscale.spv");
  frag_module = createShaderModule(frag_code);

  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );
  frag_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_FRAGMENT_BIT, frag_module
  );

  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo();
  builder.input_assembly    = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  screen_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);

  // Restore original working directory
  std::filesystem::current_path(orig_path);
}

void VulkanEngine::createDescriptorSets()
{
  auto img_count = swapchain_images.size();
  descriptor_sets.resize(img_count);

  std::vector<VkDescriptorSetLayout> layouts(img_count, descriptor_layout);
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool     = descriptor_pool;
  alloc_info.descriptorSetCount = static_cast<uint32_t>(img_count);
  alloc_info.pSetLayouts        = layouts.data();

  validation::checkVulkan(
    vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.data())
  );
}
