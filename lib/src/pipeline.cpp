#include "mimir/pipeline.hpp"

#include <spdlog/spdlog.h>
#include <dlfcn.h> // dladdr
#include <cstring> // to_string
#include <filesystem> // std::filesystem
#include <map> // std::map

#include "mimir/api.hpp"
#include "mimir/shader_types.hpp"
#include "mimir/shader.hpp"
#include "mimir/validation.hpp"

#include <spdlog/fmt/fmt.h>

// Setup the shader path so that the library can actually load them
// Hackish and Linux-only, but works for now
std::string getDefaultShaderPath()
{
    // If shaders are installed in library path, set working directory there
    Dl_info dl_info;
    dladdr((void*)getDefaultShaderPath, &dl_info);
    auto lib_pathname = dl_info.dli_fname;
    if (lib_pathname != nullptr)
    {
        std::filesystem::path lib_path(lib_pathname);
        return lib_path.parent_path().string();
    }
    else // Use executable path as working dir
    {
        auto exe_folder = std::filesystem::read_symlink("/proc/self/exe").remove_filename();
        return exe_folder;
    }
}

namespace mimir
{

VkPipelineDepthStencilStateCreateInfo getDepthInfo()
{
    // TODO: Decide when to apply depth testing
    bool use_depth = true; //(domain == DomainType::Domain3D);
    bool depth_test = use_depth;
    bool depth_write = use_depth;

    // Stencil test is currently disabled, but set its parameters explicitly just in case
    VkStencilOpState default_stencil{
        .failOp      = VK_STENCIL_OP_KEEP,
        .passOp      = VK_STENCIL_OP_KEEP,
        .depthFailOp = VK_STENCIL_OP_KEEP,
        .compareOp   = VK_COMPARE_OP_NEVER,
        .compareMask = 0,
        .writeMask   = 0,
        .reference   = 0,
    };

    VkCompareOp compare_op = VK_COMPARE_OP_GREATER_OR_EQUAL;
    return VkPipelineDepthStencilStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0, // For additional depth/stencil state info
        .depthTestEnable       = depth_test ? VK_TRUE : VK_FALSE,
        .depthWriteEnable      = depth_write ? VK_TRUE : VK_FALSE,
        .depthCompareOp        = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable     = VK_FALSE,
        .front                 = default_stencil,
        .back                  = default_stencil,
        .minDepthBounds        = 0.f,
        .maxDepthBounds        = 1.f,
    };
}

PipelineBuilder PipelineBuilder::make(VkPipelineLayout layout, VkExtent2D extent)
{
    return PipelineBuilder{
        .pipeline_infos  = {},
        .pipeline_layout = layout,
        .viewport        = {0.f, 0.f, (float)extent.width, (float)extent.height, 0.f, 1.f},
        .scissor         = { {0,0}, extent },
    };
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

VkPipelineInputAssemblyStateCreateInfo getInputAssemblyInfo(ViewType type)
{
    VkPrimitiveTopology topology;
    switch (type)
    {
        // Markers require a single point position as marker centroid
        // Same as voxels
        case ViewType::Markers:
        case ViewType::Voxels:
        default:
        {
            topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            break;
        }
        // Image uses a quad formed from two triangles
        // Edges from triangle mesh uses the primitive directly
        case ViewType::Image:
        case ViewType::Edges:
        {
            topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
        }
        // Boxes need to endpoints (a line) to determine the bounding box extent
        case ViewType::Boxes:
        {
            topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            break;
        }
    }

    return VkPipelineInputAssemblyStateCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0, // Currently unused
        .topology = topology,
        .primitiveRestartEnable = VK_FALSE,
    };
}

std::string getSpecializationName(AttributeType type, AttributeDescription attr)
{
    std::string spec = fmt::format("{}{}{}",
        getAttributeType(type), getDataType(attr.format), std::to_string(attr.format.components)
    );
    if (type != AttributeType::Position && hasIndexing(attr))
    {
        spec += std::string("FromInt");
    }
    return spec;
}

ShaderCompileParams getShaderCompileParams(ViewDescription desc)
{
    ShaderCompileParams compile;
    compile.specializations = {};

    // Make a dictionary of the attributes that need specializing,
    // while keeping note of the ones that were not specialized
    std::map<AttributeType, std::string> specs{
        {AttributeType::Position, "PositionDefault"},
        {AttributeType::Color, "ColorDefault"},
        {AttributeType::Size, "SizeDefault"},
    };

    // Get the list of specialization names
    for (auto &[type, attr] : desc.attributes) { specs[type] = getSpecializationName(type, attr); }
    for (const auto& spec : specs) { compile.specializations.push_back(spec.second); }

    // Select source code file and entry points for the view shader
    switch (desc.type)
    {
        case ViewType::Markers:
        {
            compile.module_path = "shaders/marker.slang";
            compile.entrypoints = {"vertexMain", "geometryMain", "fragmentMain"};

            // Add dimensionality specialization
            std::string marker_spec = fmt::format("Marker{}", getDomainType(desc.domain));
            compile.specializations.push_back(marker_spec);

            // Add shape specializations
            compile.specializations.push_back(getShapeStyle(desc.style));
            auto options = std::get<MarkerOptions>(desc.options);
            compile.specializations.push_back(getMarkerShape(options.shape));
            break;
        }
        case ViewType::Edges:
        {
            compile.module_path = "shaders/line.slang";
            //compile.entrypoints = {"vertexMain", "fragmentMain"};
            auto options = std::get<MeshOptions>(desc.options);
            if (options.periodic)
            {
                // Variant for periodic delaunay triangulations
                compile.entrypoints = {"vertexMain", "geometryPBCMain", "fragmentMain"};
            }
            else
            {
                compile.entrypoints = {"vertexMain", "fragmentMain"};
            }
            break;
        }
        case ViewType::Boxes:
        {
            compile.module_path = "shaders/boxes.slang";
            std::string geom_entry = fmt::format("geometryMain{}", getDomainType(desc.domain));
            compile.entrypoints = {"vertexMain", geom_entry, "fragmentMain"};
            break;
        }
        case ViewType::Voxels:
        {
            compile.module_path = "shaders/voxel.slang";
            std::string geom_entry = fmt::format("geometryMain{}", getDomainType(desc.domain));
            compile.entrypoints = {"vertexMain", geom_entry, "fragmentMain"};
            break;
        }
        case ViewType::Image:
        {
            // Do not use specializations in texture shaders for now
            compile.specializations.clear();
            compile.module_path = "shaders/texture.slang";
            compile.entrypoints = {"vertex2dMain", "frag2d_Char4"};
            break;
        }
        default:
        {
            spdlog::error("Unimplemented shader for view type {}", getViewType(desc.type));
        }
    }
    return compile;
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
            .pVertexBindingDescriptions      = bindings.data(),
            .vertexAttributeDescriptionCount = (uint32_t)attributes.size(),
            .pVertexAttributeDescriptions    = attributes.data(),
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
    validation::checkVulkan(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE,
        create_infos.size(), create_infos.data(), nullptr, pipelines.data())
    );

    // Delete shader modules after using them
    for (auto& info : pipeline_infos)
    {
        for (auto& stage : info.shader_stages)
        {
            vkDestroyShaderModule(device, stage.module, nullptr);
        }
    }
    pipeline_infos.clear();
    return pipelines;
}

uint32_t getLocationIndex(AttributeType type, int attr_idx)
{
    constexpr uint32_t max_attr_idx = 4;
    return attr_idx == 0? static_cast<uint32_t>(type) : max_attr_idx + attr_idx;
}

VertexDescription getVertexDescription(const ViewDescription desc)
{
    auto attr_count = desc.attributes.size();
    VertexDescription vert;
    vert.binding.reserve(attr_count);
    vert.attribute.reserve(attr_count);

    spdlog::trace("Adding vertex description");
    uint32_t binding = 0;
    for (auto &[type, attr] : desc.attributes)
    {
        if (type == AttributeType::Position && desc.type == ViewType::Image)
        {
            spdlog::trace("Using image vertex description");
            vert.binding.push_back(VkVertexInputBindingDescription{
                .binding   = 0,
                .stride    = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
            });
            vert.attribute.push_back(VkVertexInputAttributeDescription{
                .location = 0,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,
                .offset   = offsetof(Vertex, pos)
            });
            vert.attribute.push_back(VkVertexInputAttributeDescription{
                .location = 1,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32_SFLOAT,
                .offset   = offsetof(Vertex, uv)
            });
            return vert;
        }
        else if (type == AttributeType::Position || !hasIndexing(attr))
        {
            spdlog::trace("Adding input binding for {} attribute (binding = {}, stride = {})",
                getAttributeType(type), binding, attr.format.getSize()
            );
            vert.binding.push_back(VkVertexInputBindingDescription{
                .binding   = binding,
                .stride    = attr.format.getSize(), // sizeof(Vertex)
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            });

            vert.attribute.push_back(VkVertexInputAttributeDescription{
                .location = static_cast<uint32_t>(type),
                .binding  = binding,
                .format   = getVulkanFormat(attr.format),
                .offset   = 0,
            });
            binding++;
        }
        else
        {
            spdlog::trace("Adding index binding for {} attribute (binding = {}, stride = {})",
                getAttributeType(type), binding, attr.indexing.index_size
            );
            vert.binding.push_back(VkVertexInputBindingDescription{
                .binding   = binding,
                .stride    = attr.indexing.index_size,
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            });

            vert.attribute.push_back(VkVertexInputAttributeDescription{
                .location = static_cast<uint32_t>(type),
                .binding  = binding,
                .format   = VK_FORMAT_R32_SINT,
                .offset   = 0,
            });
            binding++;
        }
    }
    return vert;
}

uint32_t PipelineBuilder::addPipeline(const ViewDescription params, VkDevice device)
{
    auto orig_path   = std::filesystem::current_path();
    auto shader_path = getDefaultShaderPath();
    spdlog::debug("Original path: {}, shader path: {}", orig_path.string(), shader_path);
    std::filesystem::current_path(shader_path);

    auto shader_builder = ShaderBuilder::make();
    auto compile_params = getShaderCompileParams(params);
    //auto ext_shaders = params.options.external_shaders;

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    // if (ext_shaders.empty())
    // {
        spdlog::debug("Compiling slang shaders...");
        stages = shader_builder.compileModule(device, compile_params);
    // }
    // else
    // {
    //     spdlog::trace("Loading external shaders");
    //     stages = shader_builder.loadExternalShaders(device, ext_shaders);
    // }

    VkPipelineColorBlendAttachmentState color_blend{
        .blendEnable         = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
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
        .input_assembly         = getInputAssemblyInfo(params.type),
        .rasterizer             = getRasterizationInfo(params.type),
        .depth_stencil          = getDepthInfo(),
        .color_blend_attachment = color_blend,
        .multisampling          = multisampling,
    };

    // Restore original working directory
    std::filesystem::current_path(orig_path);

    pipeline_infos.push_back(info);
    return pipeline_infos.size() - 1;
}

} // namespace mimir