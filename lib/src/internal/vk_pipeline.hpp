#pragma once

#include "shader.hpp"

namespace mimir
{

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
    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    VkPipelineColorBlendAttachmentState color_blend_attachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
};

struct PipelineBuilder
{
    ShaderBuilder shader_builder;
    std::vector<PipelineInfo> pipeline_infos;
    VkPipelineLayout pipeline_layout;
    VkViewport viewport;
    VkRect2D scissor;

    PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent);
    uint32_t addPipeline(const ViewParams2 params, VkDevice device);
    std::vector<VkPipeline> createPipelines(VkDevice device, VkRenderPass pass);
};

} // namespace mimir