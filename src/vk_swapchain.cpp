#include "cudaview/vk_engine.hpp"
#include "validation.hpp"
#include "io.hpp"

#include <algorithm> // std::clamp

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
  vkDestroyPipelineLayout(device, screen_layout, nullptr);
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
}

void VulkanEngine::recreateSwapchain()
{
  vkDeviceWaitIdle(device);

  cleanupSwapchain();
  initSwapchain();
  if (texture_image != VK_NULL_HANDLE)
  {
    updateDescriptorsStructured();
  }
  else
  {
    updateDescriptorsUnstructured();
  }
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
  auto swapchain_support = getSwapchainProperties(physical_device);
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

  uint32_t queue_indices[2];
  findQueueFamilies(physical_device, queue_indices[0], queue_indices[1]);

  if (queue_indices[0] != queue_indices[1])
  {
    create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices   = queue_indices;
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
    VkFramebufferCreateInfo fb_info{};
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass      = render_pass;
    fb_info.attachmentCount = 1;
    fb_info.pAttachments    = attachments;
    fb_info.width           = swapchain_extent.width;
    fb_info.height          = swapchain_extent.height;
    fb_info.layers          = 1;

    validation::checkVulkan(vkCreateFramebuffer(
      device, &fb_info, nullptr, &framebuffers[i])
    );
  }
}

void VulkanEngine::createDescriptorPool()
{
  std::array<VkDescriptorPoolSize, 2> pool_sizes{};
  pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  pool_sizes[0].descriptorCount = static_cast<uint32_t>(swapchain_images.size());
  pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  pool_sizes[1].descriptorCount = static_cast<uint32_t>(swapchain_images.size());

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
  pool_info.pPoolSizes    = pool_sizes.data();
  pool_info.maxSets       = static_cast<uint32_t>(swapchain_images.size());
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

void VulkanEngine::createGraphicsPipeline(const std::string& vertex_file,
  const std::string& fragment_file)
{
  auto vert_code   = io::readFile(vertex_file);
  auto vert_module = createShaderModule(vert_code);

  auto frag_code   = io::readFile(fragment_file);
  auto frag_module = createShaderModule(frag_code);

  VkPipelineShaderStageCreateInfo vert_info{};
  vert_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
  vert_info.module = vert_module;
  vert_info.pName  = "main"; // Entrypoint
  // Used to specify values for shader constants
  vert_info.pSpecializationInfo = nullptr;

  VkPipelineShaderStageCreateInfo frag_info{};
  frag_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_info.module = frag_module;
  frag_info.pName  = "main"; // Entrypoint
  // Used to specify values for shader constants
  frag_info.pSpecializationInfo = nullptr;

  std::vector<VkVertexInputBindingDescription> bind_desc;
  std::vector<VkVertexInputAttributeDescription> attr_desc;
  getVertexDescriptions(bind_desc, attr_desc);

  VkPipelineVertexInputStateCreateInfo vert_input_info{};
  vert_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vert_input_info.vertexBindingDescriptionCount   = (uint32_t)bind_desc.size();
  vert_input_info.pVertexBindingDescriptions      = bind_desc.data();
  vert_input_info.vertexAttributeDescriptionCount = (uint32_t)attr_desc.size();
  vert_input_info.pVertexAttributeDescriptions    = attr_desc.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  getAssemblyStateInfo(input_assembly);

  VkViewport viewport{};
  viewport.x        = 0.f;
  viewport.y        = 0.f;
  viewport.width    = static_cast<float>(swapchain_extent.width);
  viewport.height   = static_cast<float>(swapchain_extent.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swapchain_extent;

  // Combine viewport and scissor rectangle into a viewport state
  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports    = &viewport;
  viewport_state.scissorCount  = 1;
  viewport_state.pScissors     = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable        = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth               = 1.f;
  rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable         = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.f;
  rasterizer.depthBiasClamp          = 0.f;
  rasterizer.depthBiasSlopeFactor    = 0.f;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable   = VK_FALSE;
  multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading      = 1.f;
  multisampling.pSampleMask           = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable      = VK_FALSE;

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable         = VK_FALSE;
  color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
  color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable     = VK_FALSE;
  color_blending.logicOp           = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount   = 1;
  color_blending.pAttachments      = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.f;
  color_blending.blendConstants[1] = 0.f;
  color_blending.blendConstants[2] = 0.f;
  color_blending.blendConstants[3] = 0.f;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount         = 1;
  pipeline_layout_info.pSetLayouts            = &descriptor_layout;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges    = nullptr;

  validation::checkVulkan(vkCreatePipelineLayout(
    device, &pipeline_layout_info, nullptr, &pipeline_layout)
  );

  std::vector stage_infos = {vert_info, frag_info};
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount          = static_cast<uint32_t>(stage_infos.size());
  pipeline_info.pStages             = stage_infos.data();
  pipeline_info.pVertexInputState   = &vert_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState      = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState   = &multisampling;
  pipeline_info.pDepthStencilState  = nullptr;
  pipeline_info.pColorBlendState    = &color_blending;
  pipeline_info.pDynamicState       = nullptr;
  pipeline_info.layout              = pipeline_layout;
  pipeline_info.renderPass          = render_pass;
  pipeline_info.subpass             = 0;
  pipeline_info.basePipelineHandle  = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex   = -1;

  validation::checkVulkan(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
    &pipeline_info, nullptr, &graphics_pipeline)
  );

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);
}

void VulkanEngine::createTextureGraphicsPipeline(
  const std::string& vertex_file, const std::string& fragment_file)
{
  auto vert_code = io::readFile(vertex_file);
  auto vert_module = createShaderModule(vert_code);
  VkPipelineShaderStageCreateInfo vert_info{};
  vert_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
  vert_info.module = vert_module;
  vert_info.pName  = "main"; // Entrypoint
  vert_info.pSpecializationInfo = nullptr;

  auto frag_code = io::readFile(fragment_file);
  auto frag_module = createShaderModule(frag_code);
  VkPipelineShaderStageCreateInfo frag_info{};
  frag_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_info.module = frag_module;
  frag_info.pName  = "main"; // Entrypoint
  frag_info.pSpecializationInfo = nullptr;

  VkPipelineVertexInputStateCreateInfo empty_input_state{};
  empty_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  empty_input_state.vertexBindingDescriptionCount   = 0;
  empty_input_state.pVertexBindingDescriptions      = nullptr;
  empty_input_state.vertexAttributeDescriptionCount = 0;
  empty_input_state.pVertexAttributeDescriptions    = nullptr;

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.f;
  viewport.y = 0.f;
  viewport.width    = static_cast<float>(swapchain_extent.width);
  viewport.height   = static_cast<float>(swapchain_extent.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swapchain_extent;

  // Combine viewport and scissor rectangle into a viewport state
  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount       = 1;
  viewport_state.pViewports          = &viewport;
  viewport_state.scissorCount        = 1;
  viewport_state.pScissors           = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable        = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth               = 1.f;
  rasterizer.cullMode                = VK_CULL_MODE_FRONT_BIT;
  rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable         = VK_FALSE;
  rasterizer.depthBiasConstantFactor = 0.f;
  rasterizer.depthBiasClamp          = 0.f;
  rasterizer.depthBiasSlopeFactor    = 0.f;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable   = VK_FALSE;
  multisampling.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
  multisampling.minSampleShading      = 1.f;
  multisampling.pSampleMask           = nullptr;
  multisampling.alphaToCoverageEnable = VK_FALSE;
  multisampling.alphaToOneEnable      = VK_FALSE;

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable         = VK_FALSE;
  color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.colorBlendOp        = VK_BLEND_OP_ADD;
  color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable     = VK_FALSE;
  color_blending.logicOp           = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount   = 1;
  color_blending.pAttachments      = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.f;
  color_blending.blendConstants[1] = 0.f;
  color_blending.blendConstants[2] = 0.f;
  color_blending.blendConstants[3] = 0.f;

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount         = 1;
  pipeline_layout_info.pSetLayouts            = &descriptor_layout;
  pipeline_layout_info.pushConstantRangeCount = 0;
  pipeline_layout_info.pPushConstantRanges    = nullptr;

  validation::checkVulkan(vkCreatePipelineLayout(
    device, &pipeline_layout_info, nullptr, &screen_layout)
  );

  std::vector stage_infos = {vert_info, frag_info};
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount          = static_cast<uint32_t>(stage_infos.size());
  pipeline_info.pStages             = stage_infos.data();
  pipeline_info.pVertexInputState   = &empty_input_state;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState      = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState   = &multisampling;
  pipeline_info.pDepthStencilState  = nullptr;
  pipeline_info.pColorBlendState    = &color_blending;
  pipeline_info.pDynamicState       = nullptr;
  pipeline_info.layout              = screen_layout;
  pipeline_info.renderPass          = render_pass;
  pipeline_info.subpass             = 0;
  pipeline_info.basePipelineHandle  = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex   = -1;

  validation::checkVulkan(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
    &pipeline_info, nullptr, &screen_pipeline)
  );

  vkDestroyShaderModule(device, vert_module, nullptr);
  vkDestroyShaderModule(device, frag_module, nullptr);
}

void VulkanEngine::createGraphicsPipelines()
{
  createGraphicsPipeline("_out/shaders/vertex.spv", "_out/shaders/fragment.spv");
  createTextureGraphicsPipeline(
    "_out/shaders/texture_vert.spv", "_out/shaders/texture_frag.spv"
  );
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

void VulkanEngine::updateDescriptorsUnstructured()
{
  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    VkDescriptorBufferInfo mvp_info{};
    mvp_info.buffer = uniform_buffers[i];
    mvp_info.offset = 0;
    mvp_info.range  = sizeof(UniformBufferObject); // or VK_WHOLE_SIZE

    std::array<VkWriteDescriptorSet, 2> desc_writes{};
    desc_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[0].dstSet           = descriptor_sets[i];
    desc_writes[0].dstBinding       = 0;
    desc_writes[0].dstArrayElement  = 0;
    desc_writes[0].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_writes[0].descriptorCount  = 1;
    desc_writes[0].pBufferInfo      = &mvp_info;
    desc_writes[0].pImageInfo       = nullptr;
    desc_writes[0].pTexelBufferView = nullptr;

    VkDescriptorBufferInfo extent_info{};
    extent_info.buffer = uniform_buffers[i];
    extent_info.offset = sizeof(UniformBufferObject);
    extent_info.range  = sizeof(SceneParams);

    desc_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[1].dstSet           = descriptor_sets[i];
    desc_writes[1].dstBinding       = 1;
    desc_writes[1].dstArrayElement  = 0;
    desc_writes[1].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_writes[1].descriptorCount  = 1;
    desc_writes[1].pBufferInfo      = &extent_info;
    desc_writes[1].pImageInfo       = nullptr;
    desc_writes[1].pTexelBufferView = nullptr;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}

void VulkanEngine::updateDescriptorsStructured()
{
  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    std::array<VkWriteDescriptorSet, 3> desc_writes{};

    VkDescriptorBufferInfo mvp_info{};
    mvp_info.buffer = uniform_buffers[i];
    mvp_info.offset = 0;
    mvp_info.range  = sizeof(UniformBufferObject); // or VK_WHOLE_SIZE

    desc_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[0].dstSet           = descriptor_sets[i];
    desc_writes[0].dstBinding       = 0;
    desc_writes[0].dstArrayElement  = 0;
    desc_writes[0].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_writes[0].descriptorCount  = 1;
    desc_writes[0].pBufferInfo      = &mvp_info;
    desc_writes[0].pImageInfo       = nullptr;
    desc_writes[0].pTexelBufferView = nullptr;

    VkDescriptorBufferInfo extent_info{};
    extent_info.buffer = uniform_buffers[i];
    extent_info.offset = sizeof(UniformBufferObject);
    extent_info.range  = sizeof(SceneParams);

    desc_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[1].dstSet           = descriptor_sets[i];
    desc_writes[1].dstBinding       = 1;
    desc_writes[1].dstArrayElement  = 0;
    desc_writes[1].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    desc_writes[1].descriptorCount  = 1;
    desc_writes[1].pBufferInfo      = &extent_info;
    desc_writes[1].pImageInfo       = nullptr;
    desc_writes[1].pTexelBufferView = nullptr;

    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView   = texture_view;
    image_info.sampler     = texture_sampler;

    desc_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_writes[2].dstSet          = descriptor_sets[i];
    desc_writes[2].dstBinding      = 2;
    desc_writes[2].dstArrayElement = 0;
    desc_writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    desc_writes[2].descriptorCount = 1;
    desc_writes[2].pImageInfo      = &image_info;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}
