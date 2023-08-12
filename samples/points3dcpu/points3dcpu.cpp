#include <cudaview/cudaview.hpp>

#include <algorithm> // std::clamp
#include <random>
#include <span> // std::span
#include <string> // std::stoul
#include <cudaview/validation.hpp>
using namespace validation; // checkCuda

#include "pcg/pcg_random.hpp"
#include "nvmlPower.hpp"

struct RNG
{
    pcg32 pcg;
    std::normal_distribution<float> norm;
    std::uniform_real_distribution<float> unif;

    RNG(unsigned seed): unif(0.f, 1.f) { pcg.seed(seed); }
    float uniform() { return unif(pcg); }
    float normal() { return norm(pcg); }
};

void initSystem(std::span<float3> coords, RNG& rng, uint3 extent)
{
    for (auto& coord : coords)
    {
        float rx = extent.x * rng.uniform();
        float ry = extent.y * rng.uniform();
        float rz = extent.z * rng.uniform();
        coord = {rx, ry, rz};
    }
}

void integrate3d(std::span<float3> coords, RNG& rng, uint3 extent)
{
    #pragma omp parallel for
    for (int i = 0; i < coords.size(); ++i)
    {
        auto p = coords[i];
        p.x += rng.normal();
        if (p.x > extent.x) p.x = extent.x;
        if (p.x < 0) p.x = 0;
        p.y += rng.normal();
        if (p.y > extent.x) p.y = extent.y;
        if (p.y < 0) p.y = 0;
        p.z += rng.normal();
        if (p.z > extent.z) p.z = extent.z;
        if (p.z < 0) p.z = 0;
        coords[i] = p;
    }
}

int main(int argc, char *argv[])
{
    float *d_coords       = nullptr;
    unsigned block_size   = 256;
    unsigned seed         = 123456;
    uint3 extent          = {200, 200, 200};

    // Default values for this program
    int width = 1920;
    int height = 1080;
    size_t point_count = 100;
    int iter_count = 10000;
    PresentOptions present_mode = PresentOptions::Immediate;
    size_t target_fps = 0;
    bool enable_sync = true;
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) point_count = std::stoul(argv[3]);
    if (argc >= 5) iter_count = std::stoi(argv[4]);
    if (argc >= 6) present_mode = static_cast<PresentOptions>(std::stoi(argv[5]));
    if (argc >= 7) target_fps = std::stoul(argv[6]);
    if (argc >= 8) enable_sync = static_cast<bool>(std::stoi(argv[7]));

    bool display = true;
    if (width == 0 || height == 0)
    {
        width = height = 10;
        display = false;
    }

    ViewerOptions options;
    options.window_size = {width,height}; // Starting window size
    options.show_metrics = false; // Show metrics window in GUI
    options.report_period = 0; // Print relevant usage stats every N seconds
    options.enable_sync = enable_sync;
    options.present = present_mode;
    options.target_fps = target_fps;
    CudaviewEngine engine;
    engine.init(options);

    if (display)
    {
        ViewParams params;
        params.element_count = point_count;
        params.element_size = sizeof(float3);
        params.extent = extent;
        params.data_domain = DataDomain::Domain3D;
        params.resource_type = ResourceType::UnstructuredBuffer;
        params.primitive_type = PrimitiveType::Points;
        engine.createView((void**)&d_coords, params);
    }
    else // Run the simulation without display
    {
        checkCuda(cudaMalloc((void**)&d_coords, sizeof(float3) * point_count));
    }

    std::vector<float3> coords(point_count);
    RNG rng(seed);
    initSystem(coords, rng, extent);
    auto memsize = sizeof(float3) * point_count;
    checkCuda(cudaMemcpy(d_coords, coords.data(), memsize, cudaMemcpyHostToDevice));

    printf("%s,%lu,", "host", point_count);
    GPUPowerBegin("gpu", 100);
    if (display) engine.displayAsync();

    for (size_t i = 0; i < iter_count; ++i)
    {
        integrate3d(coords, rng, extent);
        if (display) engine.prepareWindow();
        checkCuda(cudaMemcpy(d_coords, coords.data(), memsize, cudaMemcpyHostToDevice));
        if (display) engine.updateWindow();
    }

    engine.showMetrics();

    // Nvml memory report
    {
        nvmlMemory_v2_t meminfo;
        meminfo.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);
        
        constexpr double gigabyte = 1024.0 * 1024.0 * 1024.0;
        double freemem = meminfo.free / gigabyte;
        double reserved = meminfo.reserved / gigabyte;
        double totalmem = meminfo.total / gigabyte;
        double usedmem = meminfo.used / gigabyte;
        printf("%lf,%lf,", freemem, usedmem);
    }

    GPUPowerEnd();

    engine.exit();
    checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
