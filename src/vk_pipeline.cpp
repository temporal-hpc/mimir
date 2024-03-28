#include "internal/vk_pipeline.hpp"

#include <cstring> // to_string
#include <fstream> // std::ifstream
#include <map> // std::map

#include <mimir/shader_types.hpp>
#include <mimir/validation.hpp>

namespace mimir
{

VkPipelineShaderStageCreateInfo shaderStageInfo(
    VkShaderStageFlagBits stage, VkShaderModule module)
{
    VkPipelineShaderStageCreateInfo info{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext  = nullptr,
        .flags  = 0,
        .stage  = stage,  // Shader stage
        .module = module, // Module containing code for this shader stage
        .pName  = "main", // Shader entry point
        .pSpecializationInfo = nullptr, // specify values for shader constants
    };
    return info;
}

VkVertexInputBindingDescription vertexBinding(
    uint32_t binding, uint32_t stride, VkVertexInputRate rate)
{
    return VkVertexInputBindingDescription{
        .binding   = binding,
        .stride    = stride,
        .inputRate = rate,
    };
}

VkVertexInputAttributeDescription vertexAttribute(
    uint32_t location, uint32_t binding, VkFormat format, uint32_t offset)
{
    VkVertexInputAttributeDescription desc{
        .location = location,
        .binding  = binding,
        .format   = format,
        .offset   = offset,
    };
    return desc;
}

// DEPRECATED: Loads a file and returns its data buffer
// Was used for loading compiled shader files, but slang made this obsolete
std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    // Use read position to determine filesize and allocate output buffer
    auto filesize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(filesize);

    file.seekg(0);
    file.read(buffer.data(), filesize);
    file.close();
    return buffer;
}

ShaderCompileParameters getShaderCompileParams(ViewParams params)
{
    ShaderCompileParameters compile;
    compile.specializations = params.options.specializations;
    // Select source code file and entry points for the view shader
    switch (params.view_type)
    {
        case ViewType::Markers:
        {
            compile.source_path = "shaders/marker.slang";
            compile.entrypoints = {"vertexMain", "geometryMain", "fragmentMain"};

            // Make a dictionary of the attributes that need specializing,
            // while keeping note of the ones that were not specialized
            std::map<AttributeType, std::string> specs{
                {AttributeType::Position, "PositionDefault"},
                {AttributeType::Color, "ColorDefault"},
                {AttributeType::Size, "SizeDefault"}
            };
            for (const auto &[attr, memory] : params.attributes)
            {
                if (attr != AttributeType::Index)
                {
                    std::string spec = getAttributeType(attr);
                    spec += getComponentType(memory.params.component_type);
                    spec += std::to_string(memory.params.channel_count);
                    specs[attr] = spec;
                }
            }
            /*if (params.domain_type == DomainType::Structured)
            {
                specs[AttributeType::Position] = "PositionFloat3";
            }*/

            // Get the list of specialization names
            for (const auto& spec : specs)
            {
                compile.specializations.push_back(spec.second);
                //printf("added spec %s\n", spec.second.c_str());
            }

            // Add dimensionality specialization
            std::string marker_spec = "Marker";
            marker_spec += getDataDomain(params.data_domain);
            compile.specializations.push_back(marker_spec);

            // Add shape specialization
            // TODO: Do it properly
            compile.specializations.push_back("DiscShape");
            break;
        }
        case ViewType::Edges:
        {
            compile.source_path = "shaders/mesh.slang";
            compile.entrypoints = {"vertexMain", "fragmentMain"};
            // Variant for pbc delaunay edges
            //compile.entrypoints = {"vertexMain", "geometryMain", "fragmentMain"};

            // Make a dictionary of the attributes that need specializing,
            // while keeping note of the ones that were not specialized
            std::map<AttributeType, std::string> specs{
                {AttributeType::Position, "PositionDefault"},
                {AttributeType::Color, "ColorDefault"}
            };
            for (const auto &[attr, memory] : params.attributes)
            {
                if (attr != AttributeType::Index)
                {
                    std::string spec = getAttributeType(attr);
                    spec += getComponentType(memory.params.component_type);
                    spec += std::to_string(memory.params.channel_count);
                    specs[attr] = spec;
                }
            }
            for (const auto& spec : specs)
            {
                compile.specializations.push_back(spec.second);
            }
            break;
        }
        case ViewType::Voxels:
        {
            compile.source_path = "shaders/voxel.slang";
            std::string geom_entry = "geometryMain";
            geom_entry += getDataDomain(params.data_domain);
            compile.entrypoints = {"vertexImplicitMain", geom_entry, "fragmentMain"};

            std::map<AttributeType, std::string> specs{
                {AttributeType::Color, "ColorDefault"}
            };
            for (const auto &[attr, memory] : params.attributes)
            {
                if (attr != AttributeType::Index)
                {
                    std::string spec = getAttributeType(attr);
                    spec += getComponentType(memory.params.component_type);
                    spec += std::to_string(memory.params.channel_count);
                    specs[attr] = spec;
                }
            }
            for (const auto& spec : specs)
            {
                compile.specializations.push_back(spec.second);
            }

            break;
        }
        case ViewType::Image:
        {
            compile.source_path = "shaders/texture.slang";
            // The texture shader needs a specialization for the way to interpret its content
            // as a fragment. If no specialization is set, use the RawColor spec.
            if (compile.specializations.empty())
            {
                compile.specializations.push_back("RawColor");
            }
            std::string vert_entry = "vertex";
            std::string frag_entry = "frag";
            if (params.data_domain == DataDomain::Domain2D)
            {
                vert_entry += "2dMain";
                frag_entry += "2d_";
            }
            else if (params.data_domain == DataDomain::Domain3D)
            {
                vert_entry += "3dMain";
                frag_entry += "3d_";
            }
            auto color_attr = params.attributes[AttributeType::Color];
            frag_entry += getComponentType(color_attr.params.component_type);
            frag_entry += std::to_string(color_attr.params.channel_count);

            compile.entrypoints = { vert_entry, frag_entry };
            break;
        }
        default:
        {
            printf("Unimplemented shader generation for view type %s\n", getViewType(params.view_type));
        }
    }
    return compile;
}

VkPipelineDepthStencilStateCreateInfo getDepthInfo([[maybe_unused]] DataDomain domain)
{
    // TODO: Decide when to apply depth testing
    bool use_depth = true; //(domain == DataDomain::Domain3D);
    bool depth_test = use_depth;
    bool depth_write = use_depth;
    VkCompareOp compare_op = VK_COMPARE_OP_LESS;
    return VkPipelineDepthStencilStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0, // For additional depth/stencil state info
        .depthTestEnable       = depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable      = depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp        = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable     = VK_FALSE,
        .front                 = {}, // TODO: Get default values for both of these
        .back                  = {},
        .minDepthBounds        = 0.f,
        .maxDepthBounds        = 1.f,
    };
}

PipelineBuilder::PipelineBuilder(VkPipelineLayout layout, VkExtent2D extent):
    pipeline_layout{layout},
    viewport{0.f, 0.f, (float)extent.width, (float)extent.height, 0.f, 1.f},
    scissor{ {0, 0}, extent }
{
    // Create global session to work with the Slang API
    validation::checkSlang(slang::createGlobalSession(global_session.writeRef()));

    slang::TargetDesc target_desc{
        .format  = SLANG_SPIRV,
        .profile = global_session->findProfile("sm_6_6"),
    };

    const char* search_paths[] = { "shaders/include" };
    slang::SessionDesc session_desc{
        .targets                 = &target_desc,
        .targetCount             = 1,
        .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
        .searchPaths             = search_paths,
        .searchPathCount         = 1,
    };

    // Obtain a compilation session that scopes compilation and code loading
    validation::checkSlang(global_session->createSession(session_desc, session.writeRef()));
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
VkShaderModule PipelineBuilder::createShaderModule(
    const std::vector<char>& code, InteropDevice *dev)
{
    VkShaderModuleCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0, // Unused
        .codeSize = code.size(),
        .pCode    = reinterpret_cast<const uint32_t*>(code.data()),
    };

    VkShaderModule module;
    validation::checkVulkan(
        vkCreateShaderModule(dev->logical_device, &info, nullptr, &module)
    );
    // TODO: Move to vulkandevice deletor queue (likely a new one)
    dev->deletors.add([=]{
        vkDestroyShaderModule(dev->logical_device, module, nullptr);
    });
    return module;
}

std::vector<VkPipelineShaderStageCreateInfo> PipelineBuilder::compileSlang(
    InteropDevice *dev, const ShaderCompileParameters& params)
{
    SlangResult result = SLANG_OK;
    Slang::ComPtr<slang::IBlob> diag = nullptr;
    // Load code from [source_path].slang as a module
    auto module = session->loadModule(params.source_path.c_str(), diag.writeRef());
    validation::checkSlang(result, diag);

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
    result = session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diag.writeRef()
    );
    validation::checkSlang(result, diag);

    if (!params.specializations.empty())
    {
        auto layout = module->getLayout();
        std::vector<slang::SpecializationArg> args;
        for (const auto& specialization : params.specializations)
        {
            slang::SpecializationArg arg{
                .kind = slang::SpecializationArg::Kind::Type,
                .type = layout->findTypeByName(specialization.c_str()),
            };
            args.push_back(arg);
        }

        Slang::ComPtr<slang::IComponentType> spec_program;
        result = program->specialize(
            args.data(), args.size(), spec_program.writeRef(), diag.writeRef()
        );
        validation::checkSlang(result, diag);
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
        result = program->getEntryPointCode(idx, 0, kernel.writeRef(), diag.writeRef());
        validation::checkSlang(result, diag);

        VkShaderModuleCreateInfo info{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext    = nullptr,
            .flags    = 0, // Unused by Vulkan API
            .codeSize = kernel->getBufferSize(),
            .pCode    = static_cast<const uint32_t*>(kernel->getBufferPointer()),
        };
        VkShaderModule shader_module;
        validation::checkVulkan(
            vkCreateShaderModule(dev->logical_device, &info, nullptr, &shader_module)
        );
        dev->deletors.add([=]{
            vkDestroyShaderModule(dev->logical_device, shader_module, nullptr);
        });
        auto shader_info = shaderStageInfo(stage, shader_module);
        compiled_stages.push_back(shader_info);
    }
    return compiled_stages;
}

std::vector<VkPipelineShaderStageCreateInfo> PipelineBuilder::loadExternalShaders(
    InteropDevice *dev, const std::vector<ShaderInfo> shaders)
{
    std::vector<VkPipelineShaderStageCreateInfo> compiled_stages;
    for (const auto& info : shaders)
    {
        auto shader_module = createShaderModule(readFile(info.filepath), dev);
        auto shader_info = shaderStageInfo(info.stage, shader_module);
        compiled_stages.push_back(shader_info);
    }
    return compiled_stages;
}

VkPipelineRasterizationStateCreateInfo getRasterizationInfo(ViewType type)
{
    auto poly_mode = (type == ViewType::Edges)? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    return VkPipelineRasterizationStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags                   = 0, // Currently unused
        .depthClampEnable        = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode             = poly_mode,
        .cullMode                = VK_CULL_MODE_NONE,
        .frontFace               = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable         = VK_FALSE,
        .depthBiasConstantFactor = 0.f,
        .depthBiasClamp          = 0.f,
        .depthBiasSlopeFactor    = 0.f,
        .lineWidth               = 1.f,
    };
}

VkPipelineInputAssemblyStateCreateInfo getAssemblyInfo(ViewType view_type)
{
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    if (view_type == ViewType::Image || view_type == ViewType::Edges)
    {
        topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    return VkPipelineInputAssemblyStateCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0, // Currently unused
        .topology = topology,
        .primitiveRestartEnable = VK_FALSE,
    };
}

VertexDescription getVertexDescription(const ViewParams params)
{
    VertexDescription desc;
    uint32_t binding = 0;

    // TODO: This is not a special case, just a float3 coord
    if (params.view_type == ViewType::Voxels)
    {
        uint32_t location = static_cast<uint32_t>(AttributeType::Position);
        desc.binding.push_back(vertexBinding(
            binding, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX)
        );
        desc.attribute.push_back(
            vertexAttribute(location, binding, VK_FORMAT_R32G32B32_SFLOAT, 0)
        );
        binding++;
    }
    else if (params.view_type == ViewType::Image)
    {
        desc.binding.push_back(vertexBinding(
            0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX)
        );
        desc.attribute.push_back(vertexAttribute(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)
        ));
        desc.attribute.push_back(vertexAttribute(
            1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)
        ));
        return desc;
    }

    for (const auto &[attr, memory] : params.attributes)
    {
        auto& mem_params = memory.params;
        uint32_t location = static_cast<uint32_t>(attr);
        auto stride = getBytesize(mem_params.component_type, mem_params.channel_count);
        auto format = getDataFormat(mem_params.component_type, mem_params.channel_count);
        switch (attr)
        {
            // Ignore index attributes, as they are bound directly at the draw command
            case AttributeType::Position:
            case AttributeType::Color:
            case AttributeType::Size:
            {
                desc.binding.push_back(vertexBinding(
                    binding, stride, VK_VERTEX_INPUT_RATE_VERTEX
                ));
                desc.attribute.push_back(vertexAttribute(
                    location, binding, format, 0
                ));
                binding++;
                break;
            }
            default: break;
        }
    }

    return desc;
}

uint32_t PipelineBuilder::addPipeline(const ViewParams params, InteropDevice *dev)
{
    auto compile_params = getShaderCompileParams(params);
    auto ext_shaders = params.options.external_shaders;

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    if (!ext_shaders.empty()) {
        //printf("Loading external shaders\n");
        stages = loadExternalShaders(dev, ext_shaders);
    }
    else
    {
        //printf("Compiling slang shaders\n");
        stages = compileSlang(dev, compile_params);
    }

    VkPipelineColorBlendAttachmentState color_blend{
        .blendEnable         = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp        = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp        = VK_BLEND_OP_ADD,
        .colorWriteMask      =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext                 = nullptr,
        .flags                 = 0, // Currently unused
        .rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable   = VK_FALSE,
        .minSampleShading      = 1.f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable      = VK_FALSE,
    };

    PipelineInfo info{
        .shader_stages          = stages,
        .vertex_input_info      = getVertexDescription(params),
        .input_assembly         = getAssemblyInfo(params.view_type),
        .rasterizer             = getRasterizationInfo(params.view_type),
        .depth_stencil          = getDepthInfo(params.data_domain),
        .color_blend_attachment = color_blend,
        .multisampling          = multisampling,
    };

    pipeline_infos.push_back(info);
    return pipeline_infos.size() - 1;
}

std::vector<VkPipeline> PipelineBuilder::createPipelines(
    VkDevice device, VkRenderPass pass)
{
    // Combine viewport and scissor rectangle into a viewport state
    VkPipelineViewportStateCreateInfo viewport_state{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = 0, // Unused
        .viewportCount = 1,
        .pViewports    = &viewport,
        .scissorCount  = 1,
        .pScissors     = &scissor,
    };

    std::vector<VkPipelineColorBlendStateCreateInfo> color_states;
    color_states.reserve(pipeline_infos.size());

    std::vector<VkPipelineVertexInputStateCreateInfo> vertex_states;
    vertex_states.reserve(pipeline_infos.size());

    std::vector<VkGraphicsPipelineCreateInfo> create_infos;
    create_infos.reserve(pipeline_infos.size());

    for (auto& info : pipeline_infos)
    {
        // Write to color attachment with no actual blending being done
        color_states.push_back(VkPipelineColorBlendStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext           = nullptr,
            .flags           = 0, // Can be VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_ARM
            .logicOpEnable   = VK_FALSE,
            .logicOp         = VK_LOGIC_OP_NO_OP,
            .attachmentCount = 1,
            .pAttachments    = &info.color_blend_attachment,
            .blendConstants  = { 0.f, 0.f, 0.f, 0.f},
        });

        auto &bindings = info.vertex_input_info.binding;
        auto &attributes = info.vertex_input_info.attribute;
        vertex_states.push_back(VkPipelineVertexInputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0, // Currently unused
            .vertexBindingDescriptionCount   = (uint32_t)bindings.size(),
            .pVertexBindingDescriptions      = bindings.empty()? nullptr : bindings.data(),
            .vertexAttributeDescriptionCount = (uint32_t)attributes.size(),
            .pVertexAttributeDescriptions    = attributes.empty()? nullptr : attributes.data(),
        });

        // Build the pipeline
        create_infos.push_back(VkGraphicsPipelineCreateInfo{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext               = nullptr,
            .flags               = 0, // Specify how the pipeline is created
            .stageCount          = (uint32_t)info.shader_stages.size(),
            .pStages             = info.shader_stages.data(),
            .pVertexInputState   = &vertex_states.back(),
            .pInputAssemblyState = &info.input_assembly,
            .pTessellationState  = nullptr,
            .pViewportState      = &viewport_state,
            .pRasterizationState = &info.rasterizer,
            .pMultisampleState   = &info.multisampling,
            .pDepthStencilState  = &info.depth_stencil,
            .pColorBlendState    = &color_states.back(),
            .pDynamicState       = nullptr,
            .layout              = pipeline_layout,
            .renderPass          = pass,
            .subpass             = 0,
            .basePipelineHandle  = VK_NULL_HANDLE,
            .basePipelineIndex   = -1,
        });
    }

    std::vector<VkPipeline> pipelines(create_infos.size(), VK_NULL_HANDLE);
    // NOTE: 2nd parameter is pipeline cache
    validation::checkVulkan(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE,
        create_infos.size(), create_infos.data(), nullptr, pipelines.data())
    );
    pipeline_infos.clear();
    return pipelines;
}

} // namespace mimir