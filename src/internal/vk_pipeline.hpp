#pragma once

#include <slang-com-ptr.h>
#include <vulkan/vulkan.h>

#include <vector> // std::vector

#include "cudaview/engine/cudaview.hpp"
#include "cudaview/engine/vk_cudadevice.hpp"

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

struct ShaderCompileParameters
{
    std::string source_path;
    std::vector<std::string> entrypoints;
    std::vector<std::string> specializations;
};

struct PipelineBuilder
{
    std::vector<PipelineInfo> pipeline_infos;
    VkPipelineLayout pipeline_layout;
    VkViewport viewport;
    VkRect2D scissor;

    Slang::ComPtr<slang::IGlobalSession> global_session;
    Slang::ComPtr<slang::ISession> session;

    PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent);
    uint32_t addPipeline(const ViewParams params, VulkanCudaDevice *dev);
    std::vector<VkPipeline> createPipelines(VkDevice device, VkRenderPass pass);
    std::vector<VkPipelineShaderStageCreateInfo> compileSlang(
        VulkanCudaDevice *dev, const ShaderCompileParameters& params
    );
    std::vector<VkPipelineShaderStageCreateInfo> loadExternalShaders(
        VulkanCudaDevice *dev, const std::vector<ShaderInfo> shaders
    );
    VkShaderModule createShaderModule(const std::vector<char>& code, VulkanCudaDevice *dev);
};
