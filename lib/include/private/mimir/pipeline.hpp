#pragma once

#include <vulkan/vulkan.h>

#include <vector> // std::vector

#include <mimir/view.hpp>

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

// Fixed-function state builders (defined in pipeline.cpp). Exposed so a standalone pipeline (e.g. the
// Voxels-LOD variant in engine.cpp) can reuse the exact raster/depth/topology a view type renders with.
VkPipelineRasterizationStateCreateInfo getRasterizationInfo(const ViewDescription& desc);
VkPipelineDepthStencilStateCreateInfo getDepthInfo(const ViewDescription& desc);
VkPipelineInputAssemblyStateCreateInfo getInputAssemblyInfo(const ViewDescription& desc);

struct PipelineBuilder
{
    std::vector<PipelineInfo> pipeline_infos;
    VkPipelineLayout pipeline_layout;
    VkViewport viewport;
    VkRect2D scissor;

    uint32_t addPipeline(const ViewDescription desc, VkDevice device);
    std::vector<VkPipeline> createPipelines(VkDevice device, VkRenderPass pass);

    static PipelineBuilder make(VkPipelineLayout layout, VkExtent2D extent);
};

static_assert(std::is_default_constructible_v<PipelineBuilder>);
//static_assert(std::is_nothrow_default_constructible_v<PipelineBuilder>);
//static_assert(std::is_trivially_default_constructible_v<PipelineBuilder>);

} // namespace mimir