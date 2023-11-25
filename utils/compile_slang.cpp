#include <slang-com-ptr.h>

#include <filesystem> // std::filesystem
#include <fstream> // std::ofstream
#include <iostream> // std::cout
#include <experimental/source_location> // std::source_location
#include <stdexcept> // std::throw
#include <string> // std::string
#include <vector> // std::vector

using source_location = std::experimental::source_location;
namespace fs = std::filesystem;

struct ShaderCompileParameters
{
    std::string output_path;
    std::string source_path;
    std::vector<std::string> entrypoints;
    std::vector<std::string> specializations;
};

constexpr SlangResult checkSlang(SlangResult code, slang::IBlob *diag = nullptr,
    bool panic = true, source_location src = source_location::current())
{
    if (code < 0)
    {
        const char* msg = "error";
        if (diag != nullptr)
        {
            msg = static_cast<const char*>(diag->getBufferPointer());
        }
        fprintf(stderr, "Slang assertion: %s in function %s at %s(%d)\n",
            msg, src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            throw std::runtime_error("Slang failure!");
        }
    }
    return code;
}

void compileSlang(const ShaderCompileParameters& params, Slang::ComPtr<slang::ISession> session)
{
    std::cout << "Source path: " << params.source_path << "\n";

    SlangResult result = SLANG_OK;
    Slang::ComPtr<slang::IBlob> diag = nullptr;
    // Load code from [source_path].slang as a module
    auto module = session->loadModule(params.source_path.c_str(), diag.writeRef());
    checkSlang(result, diag);

    auto count = module->getDefinedEntryPointCount();
    std::cout << "Number of detected entrypoints: " << count << "\n";

    std::vector<slang::IComponentType*> components;
    components.reserve(params.entrypoints.size() + 1);
    components.push_back(module);
    // Lookup entry points by their names
    for (const auto& name : params.entrypoints)
    {
        std::cout << "  Entrypoint: " << name << "\n";
        Slang::ComPtr<slang::IEntryPoint> entrypoint = nullptr;
        module->findEntryPointByName(name.c_str(), entrypoint.writeRef());
        if (entrypoint != nullptr) components.push_back(entrypoint);
    }
    Slang::ComPtr<slang::IComponentType> program = nullptr;
    result = session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diag.writeRef()
    );
    checkSlang(result, diag);

    if (!params.specializations.empty())
    {
        std::vector<slang::SpecializationArg> args;
        for (const auto& specialization : params.specializations)
        {
        auto spec_type = module->getLayout()->findTypeByName(specialization.c_str());
        slang::SpecializationArg arg;
        arg.kind = slang::SpecializationArg::Kind::Type;
        arg.type = spec_type;
        args.push_back(arg);
        }

        Slang::ComPtr<slang::IComponentType> spec_program;
        result = program->specialize(
            args.data(), args.size(), spec_program.writeRef(), diag.writeRef()
        );
        checkSlang(result, diag);
        program = spec_program;
    }

    fs::path in_path{params.source_path};
    fs::path out_root{params.output_path};
    auto root = in_path.stem().string();
    auto layout = program->getLayout();
    for (unsigned idx = 0; idx < layout->getEntryPointCount(); ++idx)
    {
        auto entrypoint = layout->getEntryPointByIndex(idx);
        diag = nullptr;
        Slang::ComPtr<slang::IBlob> kernel = nullptr;
        result = program->getEntryPointCode(idx, 0, kernel.writeRef(), diag.writeRef());
        checkSlang(result, diag);

        auto out_filename = root + "_" + entrypoint->getName() + ".spv";
        auto out_path = out_root / out_filename;
        std::cout << out_path << "\n";
        std::ofstream out(out_path, std::ios::out | std::ios::binary);
        out.write(static_cast<const char*>(kernel->getBufferPointer()), kernel->getBufferSize());
    }
}

int main()
{
    // Create global session to work with the Slang API
    Slang::ComPtr<slang::IGlobalSession> global_session;
    checkSlang(slang::createGlobalSession(global_session.writeRef()));

    slang::TargetDesc target_desc{};
    target_desc.format = SLANG_SPIRV;
    target_desc.profile = global_session->findProfile("sm_6_6");
    const char* search_paths[] = { "samples/shaders/include" };
    slang::SessionDesc session_desc{};
    session_desc.targets = &target_desc;
    session_desc.targetCount = 1;
    session_desc.searchPaths = search_paths;
    session_desc.searchPathCount = 1;
    session_desc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    //Slang::ComPtr<slang::ICompileRequest> request;
    //checkSlang(global_session->createCompileRequest(request.writeRef()));

    // Obtain a compilation session that scopes compilation and code loading
    Slang::ComPtr<slang::ISession> session;
    checkSlang(global_session->createSession(session_desc, session.writeRef()));

    ShaderCompileParameters params;
    params.output_path = "samples/shaders";
    params.source_path = "samples/shaders/voxel.slang";
    params.entrypoints = {"vertexImplicitMain", "geometryMain", "fragmentMain"};
    compileSlang(params, session);

    params.source_path = "samples/shaders/marker.slang";
    params.entrypoints = {"vertexMain", "geometryMain", "fragmentMain"};
    params.specializations = { "PositionDouble2", "ColorInt1", "SizeDefault" };
    compileSlang(params, session);

    params.source_path = "samples/shaders/texture.slang";
    params.entrypoints = {"vertex2dMain", "frag2d_Float4"};
    params.specializations = { "RawColor" };
    compileSlang(params, session);

    params.source_path = "samples/shaders/texture.slang";
    params.entrypoints = {"vertex3dMain", "frag3d_Float1"};
    params.specializations = { "RawColor" };
    compileSlang(params, session);

    return 0;
}