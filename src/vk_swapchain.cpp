#include "cudaview/vk_engine.hpp"
#include "cudaview/io.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/vk_pipeline.hpp"
#include "internal/vk_properties.hpp"
#include "internal/validation.hpp"

#include <filesystem> // std::filesystem

#include "cudaview/vk_types.hpp"

void VulkanEngine::cleanupSwapchain()
{
  vkDestroyBuffer(device, uniform_buffer, nullptr);
  vkFreeMemory(device, ubo_memory, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  vkFreeCommandBuffers(device, command_pool,
    static_cast<uint32_t>(command_buffers.size()), command_buffers.data()
  );
  vkDestroyPipeline(device, point2d_pipeline, nullptr);
  vkDestroyPipeline(device, point3d_pipeline, nullptr);
  vkDestroyPipeline(device, mesh2d_pipeline, nullptr);
  vkDestroyPipeline(device, mesh3d_pipeline, nullptr);
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
  createSwapchain();
  createImageViews();
  createRenderPass();
  createGraphicsPipelines();
  createFramebuffers();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  updateDescriptorSets();
}

void VulkanEngine::recreateSwapchain()
{
  vkDeviceWaitIdle(device);

  cleanupSwapchain();
  initSwapchain();
}

void VulkanEngine::createSwapchain()
{
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  VkExtent2D win_ext{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  auto swapchain_support = props::getSwapchainProperties(physical_device, surface);
  auto surface_format = props::chooseSwapSurfaceFormat(swapchain_support.formats);
  auto present_mode = props::chooseSwapPresentMode(swapchain_support.present_modes);
  auto extent = props::chooseSwapExtent(swapchain_support.capabilities, win_ext);

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
  VkDescriptorPoolSize pool_sizes[] =
  {
    { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
    { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 }, // swapchain_images.size();
    { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 }, // swapchain_images.size();
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
    { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
    { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
    { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
  };

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets       = 1000; //swapchain_images.size();
  pool_info.poolSizeCount = std::size(pool_sizes);
  pool_info.pPoolSizes    = pool_sizes;

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
  auto orig_path = std::filesystem::current_path();
  std::filesystem::current_path(shader_path);
  std::cout << shader_path << "\n";

  auto vert_code   = io::readFile("shaders/unstructured/particle_pos_2d.spv");
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
  getVertexDescriptions2d(bind_desc, attr_desc);

  PipelineBuilder builder;
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  builder.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  builder.viewport.x        = 0.f;
  builder.viewport.y        = 0.f;
  builder.viewport.width    = static_cast<float>(swapchain_extent.width);
  builder.viewport.height   = static_cast<float>(swapchain_extent.height);
  builder.viewport.minDepth = 0.f;
  builder.viewport.maxDepth = 1.f;
  builder.scissor.offset    = {0, 0};
  builder.scissor.extent    = swapchain_extent;
  builder.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  builder.multisampling     = vkinit::multisamplingStateCreateInfo();
  builder.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.pipeline_layout   = pipeline_layout;
  point2d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);

  vert_code   = io::readFile("shaders/unstructured/particle_pos_3d.spv");
  vert_module = createShaderModule(vert_code);

  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );

  getVertexDescriptions3d(bind_desc, attr_desc);
  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  point3d_pipeline = builder.buildPipeline(device, render_pass);

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

  vert_code   = io::readFile("shaders/unstructured/wireframe_vertex_2d.spv");
  vert_module = createShaderModule(vert_code);

  frag_code   = io::readFile("shaders/unstructured/wireframe_fragment.spv");
  frag_module = createShaderModule(frag_code);

  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );
  frag_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_FRAGMENT_BIT, frag_module
  );

  getVertexDescriptions2d(bind_desc, attr_desc);
  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  builder.vertex_input_info = vkinit::vertexInputStateCreateInfo(bind_desc, attr_desc);
  builder.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  builder.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_LINE);
  mesh2d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);

  vert_code   = io::readFile("shaders/unstructured/wireframe_vertex_3d.spv");
  vert_module = createShaderModule(vert_code);
  vert_info = vkinit::pipelineShaderStageCreateInfo(
    VK_SHADER_STAGE_VERTEX_BIT, vert_module
  );

  getVertexDescriptions3d(bind_desc, attr_desc);
  builder.shader_stages.clear();
  builder.shader_stages.push_back(vert_info);
  builder.shader_stages.push_back(frag_info);
  mesh3d_pipeline = builder.buildPipeline(device, render_pass);

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);

  // Restore original working directory
  std::filesystem::current_path(orig_path);
}

void VulkanEngine::getVertexDescriptions2d(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(float2);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(1);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
  attr_desc[0].offset = 0;
}

void VulkanEngine::getVertexDescriptions3d(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(float3);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(1);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attr_desc[0].offset = 0;
}
