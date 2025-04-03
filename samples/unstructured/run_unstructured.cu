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
    unsigned point_count  = 100;
    unsigned iter_count   = 10000;
    double2 *d_coords     = nullptr;
    double *d_sizes       = nullptr;
    curandState *d_states = nullptr;
    int2 extent           = {200, 200};
    unsigned block_size   = 256;
    unsigned grid_size    = (point_count + block_size - 1) / block_size;
    unsigned seed         = 123456;

    if (argc >= 2) point_count = std::stoul(argv[1]);
    if (argc >= 3) iter_count  = std::stoul(argv[2]);

    // Initialize engine
    ViewerOptions options;
    options.window.size  = {1920,1080}; // Starting window size
    options.present.mode = PresentMode::Immediate;

    InstanceHandle engine = nullptr;
    createEngine(options, &engine);

    AllocHandle points, sizes;
    allocLinear(engine, (void**)&d_coords, sizeof(double2) * point_count, &points);
    allocLinear(engine, (void**)&d_sizes, sizeof(double) * point_count, &sizes);

    ViewHandle view = nullptr;
    ViewDescription desc{
        .type   = ViewType::Markers,
        .options     = MarkerOptions{
            .shape     = MarkerOptions::Shape::Disc,
        },
        .domain = DomainType::Domain2D,
        .attributes  = {
            { AttributeType::Position, {
                .source = points,
                .size   = point_count,
                .format = FormatDescription::make<double2>(),
            }},
            // { AttributeType::Size, {
            //     .source     = sizes,
            //     .size       = point_count,
            //     .format     = FormatDescription::make<double>(),
            // }},
        },
        .layout = Layout::make(point_count),
        .style  = ShapeStyle::Filled,
        .default_size = 10.f,
        .position = {-10.f, -10.f, 0.f},
        .scale    = { 0.1f, 0.1f, 0.1f },
    };
    createView(engine, &desc, &view);

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
        integrate2d<<< grid_size, block_size >>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
    };
    // Start rendering loop with the above function
    setCameraPosition(engine, {0.f, 0.f, -20.f});
    display(engine, cuda_call, iter_count);

    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_sizes));
    destroyEngine(engine);

    return EXIT_SUCCESS;
}
