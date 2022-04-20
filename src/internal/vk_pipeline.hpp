#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

struct VertexDescription
{
  std::vector<VkVertexInputBindingDescription> binding;
  std::vector<VkVertexInputAttributeDescription> attribute;
};

struct PipelineInfo
{
  std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
  VertexDescription vertex_input_info;
  VkPipelineInputAssemblyStateCreateInfo input_assembly;

  VkPipelineRasterizationStateCreateInfo rasterizer;
  VkPipelineColorBlendAttachmentState color_blend_attachment;
  VkPipelineMultisampleStateCreateInfo multisampling;
};

struct PipelineBuilder
{
  std::vector<PipelineInfo> pipeline_infos;
  VkPipelineLayout pipeline_layout;
  VkViewport viewport;
  VkRect2D scissor;

  PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent);
  uint32_t addPipelineInfo(PipelineInfo info);
  std::vector<VkPipeline> createPipelines(VkDevice device, VkRenderPass pass);
};

VertexDescription getVertexDescriptions2d();
VertexDescription getVertexDescriptions3d();
VertexDescription getVertexDescriptionsVert();
