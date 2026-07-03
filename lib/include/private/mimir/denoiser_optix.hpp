#pragma once

#include <cuda_runtime_api.h>

#include <cstddef> // size_t
#include <cstdint>

namespace mimir
{

// Thin wrapper over the OptiX AI denoiser (tensor-core accelerated on RTX GPUs). It operates on
// linear CUDA device memory holding FLOAT4 pixels (RGBA, tight-packed), so the caller feeds it the
// path-traced HDR result plus optional albedo / world-normal guide buffers and receives a denoised
// HDR result. This is the preferred path-tracing denoiser when available; the engine falls back to
// the Vulkan-compute à-trous filter when OptiX is not compiled in or initialization fails.
//
// OptiX types are kept opaque (void*/CUdeviceptr stored as uint64_t) so this header stays free of
// the OptiX SDK; all OptiX calls live in denoiser_optix.cpp behind MIMIR_HAVE_OPTIX.
class OptixDenoiser
{
public:
    OptixDenoiser() = default;
    ~OptixDenoiser() { destroy(); }
    OptixDenoiser(const OptixDenoiser&) = delete;
    OptixDenoiser& operator=(const OptixDenoiser&) = delete;

    // True if the library was built with OptiX support (MIMIR_ENABLE_OPTIX + headers found).
    static bool isCompiledIn();

    // Create an HDR denoiser sized for width x height, optionally consuming albedo / normal guides.
    // Returns false (and leaves ready() false) if OptiX is unavailable or any setup step fails, so
    // the caller can fall back to the à-trous denoiser. Safe to call again after destroy().
    bool init(uint32_t width, uint32_t height, bool use_albedo, bool use_normal);

    // Denoise the FLOAT4 device buffer `color` into `out` (both width*height*16 bytes, tight-packed),
    // guided by `albedo`/`normal` when those guides were enabled at init (ignored otherwise; may be
    // null). Enqueued on `stream`. Returns false on failure.
    bool denoise(void* color, void* albedo, void* normal, void* out, cudaStream_t stream);

    void destroy();
    bool ready() const { return ready_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    bool ready_ = false;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool use_albedo_ = false;
    bool use_normal_ = false;

    // Opaque OptiX handles (OptixDeviceContext / OptixDenoiser are pointers).
    void* context_  = nullptr;
    void* denoiser_ = nullptr;
    // CUDA device allocations (stored as raw pointers from cudaMalloc).
    void* state_     = nullptr; // denoiser state
    void* scratch_   = nullptr; // scratch memory
    void* intensity_ = nullptr; // single float: average log intensity (HDR autoexposure)
    size_t state_size_   = 0;
    size_t scratch_size_ = 0;
};

} // namespace mimir
