#include "internal/vk_pipeline.hpp"

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/validation.hpp"


ShaderCompileParameters getShaderCompileParams(ViewParams view)
{
  ShaderCompileParameters params;
  params.specializations = view.options.specializations;
  if (view.resource_type == ResourceType::Texture || view.resource_type == ResourceType::TextureLinear)
  {
    params.source_path = "shaders/texture.slang";
    params.specializations.push_back("RawColor");
    // TODO: Check why the vertex entry point names are swapped here
    if (view.data_domain == DataDomain::Domain2D)
    {
      params.entrypoints = {"vertex2dMain", "fragment2dMain"};
    }
    else if (view.data_domain == DataDomain::Domain3D)
    {
      params.entrypoints = {"vertex3dMain", "fragment3dMain"};
    }
  }
  else if (view.primitive_type == PrimitiveType::Points)
  {
    params.source_path = "shaders/marker.slang";
    if (view.data_domain == DataDomain::Domain2D)
    {
      params.entrypoints = {"vertex2dMain", "geometryMain", "fragmentMain"};
    }
    else if (view.data_domain == DataDomain::Domain3D)
    {
      params.entrypoints = {"vertex3dMain", "geometryMain", "fragmentMain"};
    }
  }
  else if (view.primitive_type == PrimitiveType::Edges)
  {
    params.source_path = "shaders/mesh.slang";
    if (view.data_domain == DataDomain::Domain2D)
    {
      params.entrypoints = {"vertex2dMain", "fragmentMain"};
    }
    else if (view.data_domain == DataDomain::Domain3D)
    {
      params.entrypoints = {"vertex3dMain", "fragmentMain"};
    }    
  }
  else if (view.primitive_type == PrimitiveType::Voxels)
  {
    params.source_path = "shaders/voxel.slang";
    if (view.resource_type == ResourceType::StructuredBuffer)
    {
      params.entrypoints = {"vertexImplicitMain", "geometryMain", "fragmentMain"};
    }
    if (view.resource_type == ResourceType::Texture)
    {
      params.entrypoints = {"vertexMain", "geometryMain", "fragmentMain"};
    }    
  }
  return params;
}

VertexDescription getVertexDescription(DataDomain domain, ResourceType res_type, PrimitiveType primitive)
{
  VertexDescription desc;

  if (res_type == ResourceType::Texture || res_type == ResourceType::TextureLinear)
  {
    desc.binding.push_back(vkinit::vertexBindingDescription(
      0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX
    ));
    desc.attribute.push_back(vkinit::vertexAttributeDescription(
      0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)
    ));
    desc.attribute.push_back(vkinit::vertexAttributeDescription(
      0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)
    ));
  }
  else
  {
    if (primitive == PrimitiveType::Voxels)
    {
      desc.binding.push_back(vkinit::vertexBindingDescription(
        0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX
      ));
      desc.attribute.push_back(vkinit::vertexAttributeDescription(
        0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0
      ));
      desc.binding.push_back(vkinit::vertexBindingDescription(
        1, sizeof(int), VK_VERTEX_INPUT_RATE_VERTEX
      ));
      desc.attribute.push_back(vkinit::vertexAttributeDescription(
        1, 1, VK_FORMAT_R32_SINT, 0
      ));
    }
    else
    {
      switch (domain)
      {
        case DataDomain::Domain2D:
        {
          desc.binding.push_back(vkinit::vertexBindingDescription(
            0, sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX
          ));
          desc.attribute.push_back(vkinit::vertexAttributeDescription(
            0, 0, VK_FORMAT_R32G32_SFLOAT, 0
          ));
          break;
        }
        case DataDomain::Domain3D:
        {
          desc.binding.push_back(vkinit::vertexBindingDescription(
            0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX
          ));
          desc.attribute.push_back(vkinit::vertexAttributeDescription(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0
          ));
          break;
        }
        default: break;
      }
    }
  }
  return desc;
}

VkPipelineInputAssemblyStateCreateInfo getAssemblyInfo(ResourceType res_type, PrimitiveType primitive)
{
  VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  if (res_type == ResourceType::Texture || res_type == ResourceType::TextureLinear || primitive == PrimitiveType::Edges)
  {
    topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
  return vkinit::inputAssemblyCreateInfo(topology);
}

VkPipelineRasterizationStateCreateInfo getRasterizationInfo(PrimitiveType primitive)
{
  VkPolygonMode poly_mode = VK_POLYGON_MODE_FILL;
  if (primitive == PrimitiveType::Edges)
  {
    poly_mode = VK_POLYGON_MODE_LINE;
  }
  return vkinit::rasterizationStateCreateInfo(poly_mode);
}

PipelineBuilder::PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent):
  pipeline_layout{layout},
  viewport{0.f, 0.f, (float)extent.width, (float)extent.height, 0.f, 1.f},
  scissor{ {0, 0}, extent }
{
  // Create global session to work with the Slang API
  validation::checkSlang(slang::createGlobalSession(global_session.writeRef()));

  slang::TargetDesc target_desc{};
  target_desc.format = SLANG_SPIRV;
  target_desc.profile = global_session->findProfile("sm_6_6");
  const char* search_paths[] = { "shaders/include" };
  slang::SessionDesc session_desc{};
  session_desc.targets = &target_desc;
  session_desc.targetCount = 1;
  session_desc.searchPaths = search_paths;
  session_desc.searchPathCount = 1;

  // Obtain a compilation session that scopes compilation and code loading
  validation::checkSlang(
    global_session->createSession(session_desc, session.writeRef())
  );
}

void diagnose(slang::IBlob *diag_blob)
{
  if (diag_blob != nullptr)
  {
    printf("%s", (const char*) diag_blob->getBufferPointer());
  }
}

VkShaderStageFlagBits getVulkanShaderFlag(SlangStage stage)
{
  switch (stage)
  {
    case SLANG_STAGE_VERTEX:   return VK_SHADER_STAGE_VERTEX_BIT;
    case SLANG_STAGE_GEOMETRY: return VK_SHADER_STAGE_GEOMETRY_BIT;
    case SLANG_STAGE_FRAGMENT: return VK_SHADER_STAGE_FRAGMENT_BIT;
    default:                   return VK_SHADER_STAGE_ALL_GRAPHICS;
  }
}

// Read buffer with shader bytecode and create a shader module from it
// DEPRECATED: The library now loads shaders via Slang
VkShaderModule PipelineBuilder::createShaderModule(
  const std::vector<char>& code, VulkanCudaDevice *dev)
{
  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.pNext    = nullptr;
  info.flags    = 0; // Unused
  info.codeSize = code.size();
  info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule module;
  validation::checkVulkan(
    vkCreateShaderModule(dev->logical_device, &info, nullptr, &module)
  );
  // TODO: Move to vulkandevice deletor queue (likely a new one)
  dev->deletors.pushFunction([=]{
    vkDestroyShaderModule(dev->logical_device, module, nullptr);
  });
  return module;
}

std::vector<VkPipelineShaderStageCreateInfo> PipelineBuilder::compileSlang(
  VulkanCudaDevice *dev, const ShaderCompileParameters& params)
{
  Slang::ComPtr<slang::IBlob> diag = nullptr;
  // Load code from [source_path].slang as a module
  auto module = session->loadModule(params.source_path.c_str(), diag.writeRef());
  diagnose(diag);

  std::vector<slang::IComponentType*> components;
  components.reserve(params.entrypoints.size() + 1);
  components.push_back(module);
  // Lookup entry points by their names
  for (const auto& name : params.entrypoints)
  {
    Slang::ComPtr<slang::IEntryPoint> entrypoint = nullptr;
    module->findEntryPointByName(name.c_str(), entrypoint.writeRef());
    if (entrypoint != nullptr) components.push_back(entrypoint);
  }
  Slang::ComPtr<slang::IComponentType> program = nullptr;
  validation::checkSlang(session->createCompositeComponentType(
    components.data(), components.size(), program.writeRef(), diag.writeRef())
  );
  diagnose(diag);

  if (!params.specializations.empty())
  {
    std::vector<slang::SpecializationArg> args;
    for (const auto& specialization : params.specializations)
    {
      auto spec_type = module->getLayout()->findTypeByName(specialization.c_str());
      slang::SpecializationArg arg;
      arg.kind = slang::SpecializationArg::Kind::Type;
      arg.type = spec_type;
      args.push_back(arg);
    }

    Slang::ComPtr<slang::IComponentType> spec_program;
    validation::checkSlang(program->specialize(
      args.data(), args.size(), spec_program.writeRef(), diag.writeRef())
    );
    diagnose(diag);
    program = spec_program;
  }

  auto layout = program->getLayout();
  std::vector<VkPipelineShaderStageCreateInfo> compiled_stages;
  compiled_stages.reserve(layout->getEntryPointCount());
  for (unsigned idx = 0; idx < layout->getEntryPointCount(); ++idx)
  {
    auto entrypoint = layout->getEntryPointByIndex(idx);
    auto stage = getVulkanShaderFlag(entrypoint->getStage());

    diag = nullptr;
    Slang::ComPtr<slang::IBlob> kernel = nullptr;
    validation::checkSlang(
      program->getEntryPointCode(idx, 0, kernel.writeRef(), diag.writeRef())
    );
    diagnose(diag);

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.pNext    = nullptr;
    info.flags    = 0; // Unused by Vulkan API
    info.codeSize = kernel->getBufferSize();
    info.pCode    = static_cast<const uint32_t*>(kernel->getBufferPointer());
    VkShaderModule shader_module;
    validation::checkVulkan(
      vkCreateShaderModule(dev->logical_device, &info, nullptr, &shader_module)
    );
    dev->deletors.pushFunction([=]{
      vkDestroyShaderModule(dev->logical_device, shader_module, nullptr);
    });
    auto shader_info = vkinit::pipelineShaderStageCreateInfo(stage, shader_module);
    compiled_stages.push_back(shader_info);
  }
  return compiled_stages;
}

uint32_t PipelineBuilder::addPipeline(const ViewParams params, VulkanCudaDevice *dev)
{
  PipelineInfo info;
  auto compile_params = getShaderCompileParams(params);
  auto stages = compileSlang(dev, compile_params);

  info.shader_stages = stages;
  info.vertex_input_info = getVertexDescription(
    params.data_domain, params.resource_type, params.primitive_type
  );
  info.input_assembly = getAssemblyInfo(params.resource_type, params.primitive_type);
  info.rasterizer = getRasterizationInfo(params.primitive_type);
  info.multisampling = vkinit::multisampleStateCreateInfo();
  info.color_blend_attachment = vkinit::colorBlendAttachmentState();

  pipeline_infos.push_back(info);
  return pipeline_infos.size() - 1;
}

std::vector<VkPipeline> PipelineBuilder::createPipelines(
  VkDevice device, VkRenderPass pass)
{
  // Combine viewport and scissor rectangle into a viewport state
  auto viewport_state = vkinit::viewportCreateInfo();
  viewport_state.viewportCount = 1;
  viewport_state.pViewports    = &viewport;
  viewport_state.scissorCount  = 1;
  viewport_state.pScissors     = &scissor;

  auto depth_stencil = vkinit::depthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS);

  std::vector<VkPipelineColorBlendStateCreateInfo> color_states;
  color_states.reserve(pipeline_infos.size());

  std::vector<VkPipelineVertexInputStateCreateInfo> vertex_states;
  vertex_states.reserve(pipeline_infos.size());

  std::vector<VkGraphicsPipelineCreateInfo> create_infos;
  create_infos.reserve(pipeline_infos.size());

  for (auto& info : pipeline_infos)
  {
    // Write to color attachment with no actual blending being done
    auto color_blend = vkinit::colorBlendInfo();
    color_blend.attachmentCount = 1;
    color_blend.pAttachments    = &info.color_blend_attachment;
    color_states.push_back(color_blend);

    auto input_info = vkinit::vertexInputStateCreateInfo(
      info.vertex_input_info.binding, info.vertex_input_info.attribute
    );
    vertex_states.push_back(input_info);

    // Build the pipeline
    auto create_info = vkinit::pipelineCreateInfo(pipeline_layout, pass);
    create_info.stageCount          = info.shader_stages.size();
    create_info.pStages             = info.shader_stages.data();
    create_info.pVertexInputState   = &vertex_states.back();
    create_info.pInputAssemblyState = &info.input_assembly;
    create_info.pViewportState      = &viewport_state;
    create_info.pRasterizationState = &info.rasterizer;
    create_info.pMultisampleState   = &info.multisampling;
    create_info.pDepthStencilState  = &depth_stencil;
    create_info.pColorBlendState    = &color_states.back();

    create_infos.push_back(create_info);
  }

  std::vector<VkPipeline> pipelines(create_infos.size(), VK_NULL_HANDLE);
  // NOTE: 2nd parameter is pipeline cache
  validation::checkVulkan(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE,
    create_infos.size(), create_infos.data(), nullptr, pipelines.data())
  );
  return pipelines;
}
