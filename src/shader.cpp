#include "internal/shader.hpp"

#include <fstream> // std::ifstream

#include <mimir/validation.hpp>

namespace mimir
{

ShaderBuilder::ShaderBuilder()
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

std::vector<VkPipelineShaderStageCreateInfo> ShaderBuilder::compileModule(
    VkDevice device, const ShaderCompileParams& params)
{
    SlangResult result = SLANG_OK;
    Slang::ComPtr<slang::IBlob> diag = nullptr;
    // Load code from [source_path].slang as a module
    auto module = session->loadModule(params.module_path.c_str(), diag.writeRef());
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

    std::vector<slang::SpecializationArg> args;
    if (!params.specializations.empty())
    {
        for (const auto& spec : params.specializations)
        {
            slang::SpecializationArg arg{
                .kind = slang::SpecializationArg::Kind::Type,
                .type = module->getLayout()->findTypeByName(spec.c_str()),
            };
            args.push_back(arg);
        }
    }
    Slang::ComPtr<slang::IComponentType> spec_program;
    result = program->specialize(args.data(), args.size(),
        spec_program.writeRef(), diag.writeRef()
    );
    validation::checkSlang(result, diag);

    Slang::ComPtr<slang::IComponentType> linked_program;
    result = spec_program->link(linked_program.writeRef(), diag.writeRef());
    validation::checkSlang(result, diag);

    auto layout = linked_program->getLayout();
    std::vector<VkPipelineShaderStageCreateInfo> compiled_stages;
    compiled_stages.reserve(layout->getEntryPointCount());
    for (unsigned idx = 0; idx < layout->getEntryPointCount(); ++idx)
    {
        auto entrypoint = layout->getEntryPointByIndex(idx);
        auto stage = getVulkanShaderFlag(entrypoint->getStage());

        diag = nullptr;
        Slang::ComPtr<slang::IBlob> kernel = nullptr;
        result = linked_program->getEntryPointCode(idx, 0, kernel.writeRef(), diag.writeRef());
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
            vkCreateShaderModule(device, &info, nullptr, &shader_module)
        );
        auto shader_info = shaderStageInfo(stage, shader_module);
        compiled_stages.push_back(shader_info);
    }
    return compiled_stages;
}

// Read buffer with shader bytecode and create a shader module from it
VkShaderModule createShaderModule(
    const std::vector<char>& code, VkDevice device)
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
        vkCreateShaderModule(device, &info, nullptr, &module)
    );
    return module;
}

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

std::vector<VkPipelineShaderStageCreateInfo> ShaderBuilder::loadExternalShaders(
    VkDevice device, const std::vector<ShaderInfo> shaders)
{
    std::vector<VkPipelineShaderStageCreateInfo> compiled_stages;
    for (const auto& info : shaders)
    {
        auto shader_module = createShaderModule(readFile(info.filepath), device);
        auto shader_info = shaderStageInfo(info.stage, shader_module);
        compiled_stages.push_back(shader_info);
    }
    return compiled_stages;
}

} // namespace mimir