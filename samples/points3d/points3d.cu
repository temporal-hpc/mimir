#include <cudaview/cudaview.hpp>

#include <curand_kernel.h>
#include <iostream> // std::cout
#include <fstream> // std::ofstream
#include <string> // std::stoul
#include <cudaview/validation.hpp>
using namespace validation; // checkCuda

#include "nvmlPower.hpp"

__global__ void initSystem(float *coords, size_t point_count,
    curandState *global_states, uint3 extent, unsigned seed)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform(&local_state);
        auto ry = extent.y * curand_uniform(&local_state);
        auto rz = extent.z * curand_uniform(&local_state);
        points[tidx] = {rx, ry, rz};
        global_states[tidx] = local_state;
    }
}

__global__ void integrate3d(float *coords, size_t point_count,
    curandState *global_states, uint3 extent)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        auto p = points[tidx];
        p.x += curand_normal(&local_state);
        if (p.x > extent.x) p.x = extent.x;
        if (p.x < 0) p.x = 0;
        p.y += curand_normal(&local_state);
        if (p.y > extent.x) p.y = extent.y;
        if (p.y < 0) p.y = 0;
        p.z += curand_normal(&local_state);
        if (p.z > extent.z) p.z = extent.z;
        if (p.z < 0) p.z = 0;
        points[tidx] = p;
        global_states[tidx] = local_state;
    }
}

int main(int argc, char *argv[])
{
    float *d_coords       = nullptr;
    curandState *d_states = nullptr;
    unsigned block_size   = 256;
    unsigned seed         = 123456;
    uint3 extent          = {200, 200, 200};

    // Default values for this program
    int width = 1920;
    int height = 1080;
    size_t point_count = 100;
    size_t iter_count  = 10000;
    PresentOptions present_mode = PresentOptions::Immediate;
    size_t target_fps  = 0;
    bool enable_sync   = true;
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) point_count = std::stoul(argv[3]);
    if (argc >= 5) iter_count = std::stoul(argv[4]);
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

    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
    unsigned grid_size = (point_count + block_size - 1) / block_size;
    initSystem<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent, seed);
    checkCuda(cudaDeviceSynchronize());

    printf("%d,%lu,", enable_sync, point_count);
    GPUPowerBegin("gpu", 100);
    if (display) engine.displayAsync();
    for (size_t i = 0; i < iter_count; ++i)
    {
        if (display) engine.prepareWindow();
        integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
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
        /*printf("Device memory report (GB):\n  free: %.2lf\n  reserved: %.2lf\n  total: %.2lf\n  used: %.2lf\n",
            freemem, reserved, totalmem, usedmem
        );*/
    }

    GPUPowerEnd();

    engine.exit();
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
