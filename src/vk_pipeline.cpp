#include "cudaview/engine/vk_pipeline.hpp"

VkPipeline PipelineBuilder::buildPipeline(VkDevice device, VkRenderPass pass)
{
  // Combine viewport and scissor rectangle into a viewport state
  // Only one viewport and scissor is supported at the moment
  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.pNext = nullptr;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports    = &viewport;
  viewport_state.scissorCount  = 1;
  viewport_state.pScissors     = &scissor;

  // Setup dummy color blending with no transparency
  // Write to color attachment with no actual blending being done
  VkPipelineColorBlendStateCreateInfo color_blend{};
  color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend.pNext = nullptr;
  color_blend.logicOpEnable     = VK_FALSE;
  color_blend.logicOp           = VK_LOGIC_OP_COPY;
  color_blend.attachmentCount   = 1;
  color_blend.pAttachments      = &color_blend_attachment;
  color_blend.blendConstants[0] = 0.f;
  color_blend.blendConstants[1] = 0.f;
  color_blend.blendConstants[2] = 0.f;
  color_blend.blendConstants[3] = 0.f;

  // Build the pipeline
  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.pNext = nullptr;
  pipeline_info.stageCount          = shader_stages.size();
  pipeline_info.pStages             = shader_stages.data();
  pipeline_info.pVertexInputState   = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState      = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState   = &multisampling;
  pipeline_info.pDepthStencilState  = nullptr;
  pipeline_info.pColorBlendState    = &color_blend;
  pipeline_info.pDynamicState       = nullptr;
  pipeline_info.layout              = pipeline_layout;
  pipeline_info.renderPass          = pass;
  pipeline_info.subpass             = 0;
  pipeline_info.basePipelineHandle  = VK_NULL_HANDLE;
  pipeline_info.basePipelineIndex   = -1;

  VkPipeline new_pipeline;
  auto result = vkCreateGraphicsPipelines(
    device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &new_pipeline
  );
  return (result != VK_SUCCESS)? VK_NULL_HANDLE : new_pipeline;
}
