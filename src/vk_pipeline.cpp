#include "internal/vk_pipeline.hpp"

#include "internal/vk_initializers.hpp"

VkPipeline PipelineBuilder::buildPipeline(VkDevice device, VkRenderPass pass)
{
  // Combine viewport and scissor rectangle into a viewport state
  auto viewport_state = vkinit::viewportCreateInfo();
  viewport_state.viewportCount = 1;
  viewport_state.pViewports    = &viewport;
  viewport_state.scissorCount  = 1;
  viewport_state.pScissors     = &scissor;

  // Setup dummy color blending with no transparency
  // Write to color attachment with no actual blending being done
  auto color_blend = vkinit::colorBlendInfo();
  color_blend.attachmentCount = 1;
  color_blend.pAttachments    = &color_blend_attachment;

  auto depth_stencil = vkinit::depthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS);

  // Build the pipeline
  auto pipeline_info = vkinit::pipelineCreateInfo(pipeline_layout, pass);
  pipeline_info.stageCount          = shader_stages.size();
  pipeline_info.pStages             = shader_stages.data();
  pipeline_info.pVertexInputState   = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState      = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState   = &multisampling;
  pipeline_info.pDepthStencilState  = &depth_stencil;
  pipeline_info.pColorBlendState    = &color_blend;

  VkPipeline new_pipeline;
  auto result = vkCreateGraphicsPipelines(
    device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &new_pipeline
  );
  return (result != VK_SUCCESS)? VK_NULL_HANDLE : new_pipeline;
}
