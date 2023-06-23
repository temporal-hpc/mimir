#include <cudaview/vk_engine.hpp>

#include <curand_kernel.h>

#include <string> // std::stoul
#include <cudaview/validation.hpp>
using namespace validation; // checkCuda

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
    size_t point_count = 100;
    size_t iter_count  = 10000;
    int width = 1280;
    int height = 720;
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) point_count = std::stoul(argv[3]);
    if (argc >= 5) iter_count = std::stoul(argv[4]);

    bool display = true;
    if (width == 0 || height == 0)
    {
        width = height = 10;
        display = false;
    }

    ViewerOptions options;
    options.window = {width,height}; // Starting window size
    options.show_metrics = false; // Show metrics window in GUI
    options.report_period = 30; // Print relevant usage stats every N seconds
    VulkanEngine engine;
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

    if (display) engine.displayAsync();
    for (size_t i = 0; i < iter_count; ++i)
    {
        if (i == iter_count / 2) engine.showMetrics();
        if (display) engine.prepareWindow();
        integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
        if (display) engine.updateWindow();
    }

    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
