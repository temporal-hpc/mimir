#include "internal/vk_pipeline.hpp"

#include "cudaview/vk_types.hpp"
#include "internal/vk_initializers.hpp"

PipelineBuilder::PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent):
  pipeline_layout{layout},
  viewport{0.f, 0.f, (float)extent.width, (float)extent.height, 0.f, 1.f},
  scissor{ {0, 0}, extent }
{
  pipeline_infos.reserve(6);
}

uint32_t PipelineBuilder::addPipelineInfo(PipelineInfo info)
{
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

  std::vector<VkGraphicsPipelineCreateInfo> create_infos;
  create_infos.reserve(pipeline_infos.size());
  for (auto& info : pipeline_infos)
  {
    // Write to color attachment with no actual blending being done
    auto color_blend = vkinit::colorBlendInfo();
    color_blend.attachmentCount = 1;
    color_blend.pAttachments    = &info.color_blend_attachment;

    auto depth_stencil = vkinit::depthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS);

    auto input_info = vkinit::vertexInputStateCreateInfo(
      info.vertex_input_info.binding, info.vertex_input_info.attribute
    );

    // Build the pipeline
    auto create_info = vkinit::pipelineCreateInfo(pipeline_layout, pass);
    create_info.stageCount          = info.shader_stages.size();
    create_info.pStages             = info.shader_stages.data();
    create_info.pVertexInputState   = &input_info;
    create_info.pInputAssemblyState = &info.input_assembly;
    create_info.pViewportState      = &viewport_state;
    create_info.pRasterizationState = &info.rasterizer;
    create_info.pMultisampleState   = &info.multisampling;
    create_info.pDepthStencilState  = &depth_stencil;
    create_info.pColorBlendState    = &color_blend;

    create_infos.push_back(create_info);
  }

  std::vector<VkPipeline> pipelines(create_infos.size(), VK_NULL_HANDLE);
  vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, // pipeline cache
    create_infos.size(), create_infos.data(), nullptr, pipelines.data()
  );
  return pipelines;
}

VertexDescription getVertexDescriptions2d()
{
  VertexDescription desc;
  desc.binding.push_back(
    vkinit::vertexBindingDescription(0, sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX)
  );
  desc.attribute.push_back(
    vkinit::vertexAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, 0)
  );
  return desc;
}

VertexDescription getVertexDescriptions3d()
{
  VertexDescription desc;
  desc.binding.push_back(
    vkinit::vertexBindingDescription(0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX)
  );
  desc.attribute.push_back(
    vkinit::vertexAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
  );
  return desc;
}

VertexDescription getVertexDescriptionsVert()
{
  VertexDescription desc;
  desc.binding.push_back(
    vkinit::vertexBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
  );
  desc.attribute.push_back(
    vkinit::vertexAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
  );
  desc.attribute.push_back(
    vkinit::vertexAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv))
  );
  return desc;
}
