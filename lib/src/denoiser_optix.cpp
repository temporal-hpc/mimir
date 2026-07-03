#include "mimir/denoiser_optix.hpp"

#include <spdlog/spdlog.h>

#ifdef MIMIR_HAVE_OPTIX
#include <optix.h>
#include <optix_stubs.h>
// Defines the OptiX function table exactly once for the whole library (must appear in a single TU).
#include <optix_function_table_definition.h>
#include <cuda.h> // CUdeviceptr, CUcontext

#include <algorithm> // std::max
#include <cstring>   // std::memset
#endif

namespace mimir
{

bool OptixDenoiser::isCompiledIn()
{
#ifdef MIMIR_HAVE_OPTIX
    return true;
#else
    return false;
#endif
}

#ifndef MIMIR_HAVE_OPTIX

// ---- Stub build (no OptiX): every entry point reports "unavailable" so the engine falls back. ----
bool OptixDenoiser::init(uint32_t, uint32_t, bool, bool) { return false; }
bool OptixDenoiser::denoise(void*, void*, void*, void*, cudaStream_t) { return false; }
void OptixDenoiser::destroy() {}

#else

namespace
{

// One-time optixInit() (loads the function table via the CUDA driver). Returns false if OptiX is
// present at build time but unusable at runtime (no driver / no CUDA context yet).
bool ensureOptixInit()
{
    static bool tried = false;
    static bool ok = false;
    if (!tried)
    {
        tried = true;
        OptixResult r = optixInit();
        ok = (r == OPTIX_SUCCESS);
        if (!ok) { spdlog::warn("optixInit failed ({}); using à-trous denoiser", (int)r); }
    }
    return ok;
}

void optixLog(unsigned int level, const char* tag, const char* msg, void*)
{
    spdlog::debug("[OptiX][{}] {}: {}", level, tag ? tag : "", msg ? msg : "");
}

// A tight-packed FLOAT4 image over a linear device buffer.
OptixImage2D float4Image(void* ptr, uint32_t w, uint32_t h)
{
    OptixImage2D img{};
    img.data = reinterpret_cast<CUdeviceptr>(ptr);
    img.width = w;
    img.height = h;
    img.rowStrideInBytes = w * static_cast<unsigned>(sizeof(float) * 4);
    img.pixelStrideInBytes = static_cast<unsigned>(sizeof(float) * 4);
    img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return img;
}

} // namespace

bool OptixDenoiser::init(uint32_t width, uint32_t height, bool use_albedo, bool use_normal)
{
    destroy();
    if (width == 0 || height == 0) { return false; }
    if (!ensureOptixInit()) { return false; }

    // Create an OptiX device context on the current CUDA context (0 = current).
    OptixDeviceContextOptions ctx_options{};
    ctx_options.logCallbackFunction = &optixLog;
    ctx_options.logCallbackLevel = 3;
    OptixDeviceContext context = nullptr;
    if (optixDeviceContextCreate(0 /*current CUDA context*/, &ctx_options, &context) != OPTIX_SUCCESS)
    {
        spdlog::warn("optixDeviceContextCreate failed; using à-trous denoiser");
        return false;
    }
    context_ = context;

    // Create an HDR denoiser with the requested guide layers.
    OptixDenoiserOptions options{};
    options.guideAlbedo = use_albedo ? 1u : 0u;
    options.guideNormal = use_normal ? 1u : 0u;
    ::OptixDenoiser denoiser = nullptr;
    if (optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_HDR, &options, &denoiser)
        != OPTIX_SUCCESS)
    {
        spdlog::warn("optixDenoiserCreate failed; using à-trous denoiser");
        destroy();
        return false;
    }
    denoiser_ = denoiser;

    OptixDenoiserSizes sizes{};
    if (optixDenoiserComputeMemoryResources(denoiser, width, height, &sizes) != OPTIX_SUCCESS)
    {
        spdlog::warn("optixDenoiserComputeMemoryResources failed; using à-trous denoiser");
        destroy();
        return false;
    }
    state_size_ = sizes.stateSizeInBytes;
    // Scratch must also cover the intensity computation (we use a shared scratch buffer).
    scratch_size_ = std::max(sizes.withoutOverlapScratchSizeInBytes, sizes.computeIntensitySizeInBytes);

    if (cudaMalloc(&state_, state_size_) != cudaSuccess
        || cudaMalloc(&scratch_, scratch_size_) != cudaSuccess
        || cudaMalloc(&intensity_, sizeof(float)) != cudaSuccess)
    {
        spdlog::warn("cudaMalloc for OptiX denoiser memory failed; using à-trous denoiser");
        destroy();
        return false;
    }

    if (optixDenoiserSetup(denoiser, /*stream=*/0, width, height,
            reinterpret_cast<CUdeviceptr>(state_), state_size_,
            reinterpret_cast<CUdeviceptr>(scratch_), scratch_size_) != OPTIX_SUCCESS)
    {
        spdlog::warn("optixDenoiserSetup failed; using à-trous denoiser");
        destroy();
        return false;
    }

    width_ = width;
    height_ = height;
    use_albedo_ = use_albedo;
    use_normal_ = use_normal;
    ready_ = true;
    spdlog::info("OptiX AI denoiser ready ({}x{}, albedo={}, normal={})",
        width, height, use_albedo, use_normal);
    return true;
}

bool OptixDenoiser::denoise(void* color, void* albedo, void* normal, void* out, cudaStream_t stream)
{
    if (!ready_ || color == nullptr || out == nullptr) { return false; }
    auto denoiser = static_cast<::OptixDenoiser>(denoiser_);
    CUstream cu_stream = reinterpret_cast<CUstream>(stream);

    OptixImage2D input = float4Image(color, width_, height_);

    // Autoexposure: compute the average log intensity so the HDR model tone-maps consistently.
    if (optixDenoiserComputeIntensity(denoiser, cu_stream, &input,
            reinterpret_cast<CUdeviceptr>(intensity_),
            reinterpret_cast<CUdeviceptr>(scratch_), scratch_size_) != OPTIX_SUCCESS)
    {
        return false;
    }

    OptixDenoiserGuideLayer guide{};
    if (use_albedo_ && albedo != nullptr) { guide.albedo = float4Image(albedo, width_, height_); }
    if (use_normal_ && normal != nullptr) { guide.normal = float4Image(normal, width_, height_); }

    OptixDenoiserLayer layer{};
    layer.input = input;
    layer.output = float4Image(out, width_, height_);

    OptixDenoiserParams params{};
    params.hdrIntensity = reinterpret_cast<CUdeviceptr>(intensity_);
    params.blendFactor = 0.f; // 0 = fully denoised

    OptixResult r = optixDenoiserInvoke(denoiser, cu_stream, &params,
        reinterpret_cast<CUdeviceptr>(state_), state_size_,
        &guide, &layer, /*numLayers=*/1, /*offsetX=*/0, /*offsetY=*/0,
        reinterpret_cast<CUdeviceptr>(scratch_), scratch_size_);
    if (r != OPTIX_SUCCESS)
    {
        spdlog::warn("optixDenoiserInvoke failed ({})", (int)r);
        return false;
    }
    return true;
}

void OptixDenoiser::destroy()
{
    if (denoiser_) { optixDenoiserDestroy(static_cast<::OptixDenoiser>(denoiser_)); denoiser_ = nullptr; }
    if (context_)  { optixDeviceContextDestroy(static_cast<OptixDeviceContext>(context_)); context_ = nullptr; }
    if (state_)     { cudaFree(state_);     state_ = nullptr; }
    if (scratch_)   { cudaFree(scratch_);   scratch_ = nullptr; }
    if (intensity_) { cudaFree(intensity_); intensity_ = nullptr; }
    state_size_ = 0;
    scratch_size_ = 0;
    ready_ = false;
    width_ = height_ = 0;
}

#endif // MIMIR_HAVE_OPTIX

} // namespace mimir
