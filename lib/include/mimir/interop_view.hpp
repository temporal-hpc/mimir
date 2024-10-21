#pragma once

//#include "vk_mem_alloc.h"

#include <map> // std::map
#include <memory> // std::shared_ptr
#include <string> // std::string
#include <vector> // std::vector

namespace mimir
{

struct ShaderInfo
{
    std::string filepath;
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_ALL_GRAPHICS;
};

struct ViewOptions
{
    // Customizable name for the view
    std::string name;
    // Flag indicating if this view should be displayed or not
    bool visible = true;
    // Default primitive color if no per-instance color is set
    float4 default_color{0.f,0.f,0.f,1.f};
    // Default primitive size if no per-instance size is set
    float default_size = 10.f;
    // Index of instance inside view
    int scenario_index = 0;
    // For moving through the different slices in a 3D texture
    float depth = 0.1f;
    // External alternate shaders for use in this view
    std::vector<ShaderInfo> external_shaders;
    // For specializing slang shaders associated to this view
    std::vector<std::string> specializations;
    int custom_val = 0;
};

using AttributeDict = std::map<AttributeType, AttributeParams>;

struct ViewParams
{
    size_t element_count = 0;
    uint3 extent = {1, 1, 1};
    DomainType data_domain;
    ViewType view_type;
    ViewOptions options;
    AttributeDict attributes;
    IndexingParams indexing;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> sizes;
};

} // namespace mimir