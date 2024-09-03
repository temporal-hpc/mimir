#include <curand_kernel.h>
#include <iostream> // std::cerr
#include <string> // std::stoul

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

__global__
void initSystem(double2 *coords, double *sizes, size_t point_count,
    curandState *global_states, int2 extent, unsigned seed)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform_double(&local_state);
        auto ry = extent.y * curand_uniform_double(&local_state);
        // Generate a point size up to 10;
        double2 p{rx, ry};
        coords[tidx] = p;
        sizes[tidx]  = 10 * curand_uniform_double(&local_state);
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
    double2 *d_coords     = nullptr;
    double *d_sizes       = nullptr;
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
        options.window.size  = {1920,1080}; // Starting window size
        options.present.mode = PresentMode::VSync;
        MimirEngine engine;
        engine.init(options);

        auto points = engine.allocLinear((void**)&d_coords, sizeof(double2) * point_count);
        auto sizes  = engine.allocLinear((void**)&d_sizes, sizeof(double) * point_count);

        ViewParams2 params;
        params.element_count = point_count;
        params.extent        = {200, 200, 1};
        params.data_domain   = DataDomain::Domain2D;
        params.view_type     = ViewType::Markers;
        params.options.default_size = 20.f;
        params.attributes[AttributeType::Position] = {
            .allocation = points,
            .format     = { .type = DataType::float64, .components = 2 },
        };
        params.attributes[AttributeType::Size] = {
            .allocation = sizes,
            .format     = { .type = DataType::float64, .components = 1 },
        };
        engine.createView(params);

        // params.options.external_shaders = {
        //     {"shaders/marker_vertexMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        //     {"shaders/marker_geometryMain.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
        //     {"shaders/marker_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
        // };

        // Cannot make CUDA calls that use the target device memory before
        // registering it on the engine
        //checkCuda(cudaMalloc(&d_coords, sizeof(double2) * point_count));
        checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
        initSystem<<<grid_size, block_size>>>(
            d_coords, d_sizes, point_count, d_states, extent, seed
        );
        checkCuda(cudaDeviceSynchronize());

        // Set up the cuda code that updates the view buffer as a lambda function
        auto cuda_call = [&]
        {
            integrate2d<<< grid_size, block_size >>>(
                d_coords, point_count, d_states, extent
            );
            checkCuda(cudaDeviceSynchronize());
        };
        // Start rendering loop with the above function
        engine.display(cuda_call, iter_count);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_sizes));

    return EXIT_SUCCESS;
}
