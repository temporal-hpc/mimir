#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits> // std::numeric_limits

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

constexpr float max_distance = std::numeric_limits<float>::max();

__device__
float4 jumpFloodStep(float2 coord, float4 *seeds, int step_length, int2 extent)
{
    float best_dist = max_distance;
    float2 best_coord = make_float2(-1.f, -1.f);

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            int sample_x = coord.x + x * step_length;
            int sample_y = coord.y + y * step_length;
            if (sample_x >= 0 && sample_x < extent.x && sample_y >= 0 && sample_y < extent.y)
            {
                float4 seed = seeds[extent.x * sample_y + sample_x];
                float dist = hypotf(seed.x - coord.x, seed.y - coord.y);

                if ((seed.x != -1.f && seed.y != -1.f) && dist < best_dist)
                {
                    best_dist = dist;
                    best_coord = make_float2(seed.x, seed.y);
                }
            }
        }
    }
    return make_float4(best_coord.x, best_coord.y, 0.f, best_dist);
}

__global__
void kernelJfa(float4 *result, float4 *seeds, const int2 extent, int step_length)
{
    const int tx = blockDim.x * blockIdx.x + threadIdx.x;
    const int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < extent.x && ty < extent.y)
    {
        float2 coord = make_float2(tx, ty);
        float4 output = jumpFloodStep(coord, seeds, step_length, extent);
        result[extent.x * ty + tx] = output;
    }
}

__global__
void kernelDistanceTransform(float *distances, float4 *seeds, int2 extent)
{
    const int tx = blockDim.x * blockIdx.x + threadIdx.x;
    const int ty = blockDim.y * blockIdx.y + threadIdx.y;

    if (tx < extent.x && ty < extent.y)
    {
        auto grid_idx = extent.x * ty + tx;
        distances[grid_idx] = seeds[grid_idx].w / 200.f;//hypotf(extent.x, extent.y);
    }
}

void jumpFlood(float *distances, float4 *seeds[], int2 extent)
{
    dim3 threads(32, 32);
    dim3 blocks( (extent.x + threads.x - 1) / threads.x,
                 (extent.y + threads.y - 1) / threads.y );

    int out_idx = 0, in_idx = 1;
    for (int k = extent.x / 2; k > 0; k = k >> 1)
    {
        kernelJfa<<< blocks, threads >>>(seeds[out_idx], seeds[in_idx], extent, k);
        checkCuda(cudaDeviceSynchronize());
        std::swap(out_idx, in_idx);
    }
    kernelDistanceTransform<<< blocks, threads >>>(distances, seeds[in_idx], extent);
}

__global__
void kernelSetNonSeeds(float4 *seeds, int seed_count)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < seed_count)
    {
        seeds[tx] = make_float4(-1.f, -1.f, 0.f, 0.f);
    }
}

__global__
void kernelSetSeeds(float4 *seeds, float *raw_coords, int coord_count, int2 extent)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tx < coord_count)
    {
        auto coord = reinterpret_cast<float2*>(raw_coords)[tx];
        int2 point{ (int)coord.x, (int)coord.y };
        if (point.x >= 0 && point.x < extent.x && point.y >= 0 && point.y < extent.y)
        {
            seeds[extent.x * point.y + point.x] = make_float4(coord.x, coord.y, 0.f, 0.f);
        }
    }
}

void initJumpFlood(float4 *d_seeds, float *d_coords, int coord_count,
    int2 extent)
{
    dim3 threads{128};
    dim3 blocks1{ (extent.x * extent.y + threads.x - 1) / threads.x};
    dim3 blocks2{ (coord_count + threads.x - 1) / threads.x};

    kernelSetNonSeeds<<< blocks1, threads >>>(d_seeds, extent.x * extent.y);
    checkCuda(cudaDeviceSynchronize());
    kernelSetSeeds<<< blocks2, threads >>>(d_seeds, d_coords, coord_count, extent);
    checkCuda(cudaDeviceSynchronize());
}

__global__ void initSystem(float *coords, size_t particle_count,
    curandState *global_states, int2 extent, unsigned seed)
{
    auto particles = reinterpret_cast<float2*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < particle_count)
    {
        auto local_state = global_states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform(&local_state);
        auto ry = extent.y * curand_uniform(&local_state);
        float2 p{rx, ry};
        particles[tidx] = p;
        global_states[tidx] = local_state;
    }
}

__device__ float clamp(float x, float low, float high)
{
    return fmaxf(low, fminf(high, x));
}

__global__ void integrate2d(float *coords, size_t particle_count,
    curandState *global_states, int2 extent)
{
    auto particles = reinterpret_cast<float2*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < particle_count)
    {
        auto local_state = global_states[tidx];
        auto r = curand_normal2(&local_state);
        auto p = particles[tidx];
        p.x = clamp(p.x + r.x, 1e-6f, extent.x);
        p.y = clamp(p.y + r.y, 1e-6f, extent.y);
        particles[tidx] = p;
        global_states[tidx] = local_state;
    }
}

int main(int argc, char *argv[])
{
    float *d_distances    = nullptr;
    float *d_coords       = nullptr;
    float4 *d_grid[2]     = {nullptr, nullptr};
    curandState *d_states = nullptr;
    int2 extent           = {512, 512};

    unsigned seed_count = 100;
    size_t iter_count = 10000;
    if (argc >= 2) seed_count = std::stoul(argv[1]);
    if (argc >= 3) iter_count  = std::stoul(argv[2]);

    InstanceHandle instance = nullptr;
    createInstance(1920, 1080, &instance);

    AllocHandle seeds, field;
    allocLinear(instance, (void**)&d_coords, sizeof(float2) * seed_count, &seeds);
    allocLinear(instance, (void**)&d_distances, sizeof(float) * extent.x * extent.y, &field);

    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc;
    desc.layout = Layout::make(seed_count);
    desc.type   = ViewType::Markers;
    desc.domain = DomainType::Domain2D;
    desc.attributes[AttributeType::Position] = {
        .source = seeds,
        .size   = seed_count,
        .format = FormatDescription::make<float2>(),
    };
    desc.default_size  = 10.f;
    desc.default_color = {0,0,1,1};
    desc.position      = {-25.f, -25.f, 0.f};
    desc.scale         = {0.1f, 0.1f, 0.1f};
    createView(instance, &desc, &v1);

    desc.layout = Layout::make(extent.x, extent.y);
    desc.type   = ViewType::Voxels;
    desc.attributes[AttributeType::Position] = makeStructuredGrid(instance, desc.layout);
    desc.attributes[AttributeType::Color] = {
        .source = field,
        .size   = (uint)(extent.x * extent.y),
        .format = FormatDescription::make<float>(),
    };
    desc.default_size  = 1000.f;
    desc.position      = {-25.f, -25.f, 0.49f};
    createView(instance, &desc, &v2);

    setCameraPosition(instance, {0.f, 0.f, -40.f});

    //checkCuda(cudaMalloc(&_d_distances, dist_size));
    //checkCuda(cudaMalloc(&d_coords, sizeof(float2) * element_count));
    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * seed_count));

    dim3 threads{128};
    dim3 blocks { (seed_count + threads.x - 1) / threads.x};
    initSystem<<<blocks, threads>>>(d_coords, seed_count, d_states, extent, 1234);
    checkCuda(cudaDeviceSynchronize());

    // Allocate device numeric canvas
    size_t seed_sizes = sizeof(float4) * extent.x * extent.y;
    checkCuda(cudaMalloc(&d_grid[0], seed_sizes));
    checkCuda(cudaMalloc(&d_grid[1], seed_sizes));
    checkCuda(cudaDeviceSynchronize());
    initJumpFlood(d_grid[1], d_coords, seed_count, extent);

    // Start rendering loop
    auto timestep_function = [&]{
        dim3 threads{128};
        dim3 blocks { (seed_count + threads.x - 1) / threads.x};

        integrate2d<<< blocks, threads >>>(d_coords, seed_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
        initJumpFlood(d_grid[1], d_coords, seed_count, extent);
        jumpFlood(d_distances, d_grid, extent);
        checkCuda(cudaDeviceSynchronize());
    };
    display(instance, timestep_function, iter_count);

    checkCuda(cudaFree(d_grid[0]));
    checkCuda(cudaFree(d_grid[1]));
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_distances));
    checkCuda(cudaFree(d_coords));
    destroyInstance(instance);

    return EXIT_SUCCESS;
}
