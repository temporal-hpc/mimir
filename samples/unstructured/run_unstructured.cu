#include <curand_kernel.h>
#include <iostream> // std::cerr
#include <string> // std::stoul

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

__global__
void initSystem(double2 *coords, size_t point_count, curandState *global_states, int2 extent, unsigned seed)
{   
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform_double(&local_state);
        auto ry = extent.y * curand_uniform_double(&local_state);
        double2 p{rx, ry};
        coords[tidx] = p;
        global_states[tidx] = local_state;
    }
}

__global__
void integrate2d(double2 *coords, size_t point_count, curandState *states, int2 extent)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = states[tidx];
        auto r = curand_normal2_double(&local_state);
        auto p = coords[tidx];
        p.x += r.x;
        if (p.x > extent.x) p.x = extent.x;
        if (p.x < 0) p.x = 0;
        p.y += r.y;
        if (p.y > extent.y) p.y = extent.y;
        if (p.y < 0) p.y = 0;
        coords[tidx] = p;
        states[tidx] = local_state;
    }
}

int main(int argc, char *argv[])
{
    size_t point_count    = 100;
    size_t iter_count     = 10000;
    double2 *d_coords      = nullptr;
    curandState *d_states = nullptr;
    int2 extent           = {200, 200};
    unsigned block_size   = 256;
    unsigned grid_size    = (point_count + block_size - 1) / block_size;
    unsigned seed         = 123456;

    if (argc >= 2) point_count = std::stoul(argv[1]);
    if (argc >= 3) iter_count = std::stoul(argv[2]);
    try
    {
        // Initialize engine
        ViewerOptions options;
        options.window_size = {1920,1080}; // Starting window size
        options.present = PresentOptions::VSync;
        CudaviewEngine engine;
        engine.init(options);
        ViewParams params;
        params.element_count = point_count;
        params.extent        = {200, 200, 1};
        params.data_type     = DataType::Double;
        params.channel_count = 2;
        params.data_domain   = DataDomain::Domain2D;
        params.resource_type = ResourceType::Buffer;
        params.domain_type   = DomainType::Unstructured;
        params.element_type  = ElementType::Markers;
        params.options.size  = 20.f;
        params.options.external_shaders = {
            {"shaders/marker_vertexMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
            {"shaders/marker_geometryMain.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
            {"shaders/marker_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
        };
        engine.createView((void**)&d_coords, params);

        // Cannot make CUDA calls that use the target device memory before
        // registering it on the engine
        checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
        initSystem<<<grid_size, block_size>>>(
            d_coords, point_count, d_states, extent, seed
        );
        checkCuda(cudaDeviceSynchronize());

        // Start rendering loop
        engine.display([&]
        {
            integrate2d<<< grid_size, block_size >>>(
                d_coords, point_count, d_states, extent
            );
            checkCuda(cudaDeviceSynchronize());
        }, iter_count);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
