#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

struct PipelineBuilder
{
  VkPipelineLayout pipeline_layout;
  VkViewport viewport;
  VkRect2D scissor;
  
  std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
  VkPipelineVertexInputStateCreateInfo vertex_input_info;
  VkPipelineInputAssemblyStateCreateInfo input_assembly;

  VkPipelineRasterizationStateCreateInfo rasterizer;
  VkPipelineColorBlendAttachmentState color_blend_attachment;
  VkPipelineMultisampleStateCreateInfo multisampling;

  PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent);
  VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
};
