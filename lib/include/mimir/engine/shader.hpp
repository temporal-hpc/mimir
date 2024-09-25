#pragma once

#define SLANG_CUDA_ENABLE_HALF
#include <slang-com-ptr.h>
#include <vulkan/vulkan.h>

#include <string> // std::string
#include <vector> // std::vector

#include <mimir/engine/interop_view.hpp> // ShaderInfo

namespace mimir
{

struct ShaderCompileParams
{
    std::string module_path;
    std::vector<std::string> entrypoints;
    std::vector<std::string> specializations;
};

struct ShaderBuilder
{
    Slang::ComPtr<slang::IGlobalSession> global_session;
    Slang::ComPtr<slang::ISession> session;

    std::vector<VkPipelineShaderStageCreateInfo> compileModule(
        VkDevice device, const ShaderCompileParams& params
    );
    std::vector<VkPipelineShaderStageCreateInfo> loadExternalShaders(
        VkDevice device, const std::vector<ShaderInfo> shaders
    );

    static ShaderBuilder make();
};

} // namespace mimir