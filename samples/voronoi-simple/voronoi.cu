#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits> // std::numeric_limits

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

constexpr float max_distance = std::numeric_limits<float>::max();

__device__ int getLinearIndex(int x, int y, int2 extent)
{
    return extent.x * y + x;
}

__device__
float4 jumpFloodStep(float2 coord, float4 *grid, int step_length, int2 extent)
{
    float best_dist = max_distance;
    float2 best_coord = make_float2(-1.f, -1.f);
    float best_point_idx = -1.f;

    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            int sample_x = coord.x + x * step_length;
            int sample_y = coord.y + y * step_length;
            if (sample_x >= 0 && sample_x < extent.x && sample_y >= 0 && sample_y < extent.y)
            {
                int cell_idx = getLinearIndex(sample_x, sample_y, extent);
                float4 seed = grid[cell_idx];
                float dist = hypotf(seed.x - coord.x, seed.y - coord.y);

                if ((seed.x != -1.f && seed.y != -1.f) && dist < best_dist)
                {
                    best_dist = dist;
                    best_coord = make_float2(seed.x, seed.y);
                    best_point_idx = seed.w;
                }
            }
        }
    }
    return {best_coord.x, best_coord.y, best_dist, best_point_idx };
}

__global__
void kernelJfa(float4 *result, float4 *grid, const int2 extent, int step_length)
{
    const int tx = blockDim.x * blockIdx.x + threadIdx.x;
    const int ty = blockDim.y * blockIdx.y + threadIdx.y;
    if (tx < extent.x && ty < extent.y)
    {
        float2 coord = make_float2(tx + .5f, ty + .5f);
        float4 output = jumpFloodStep(coord, grid, step_length, extent);
        result[extent.x * ty + tx] = output;
    }
}

__global__
void kernelWriteResult(float *vd_dists, float4 *vd_colors, float4 *grid, float3 *seed_colors, int2 extent)
{
    const int tx = blockDim.x * blockIdx.x + threadIdx.x;
    const int ty = blockDim.y * blockIdx.y + threadIdx.y;

    if (tx < extent.x && ty < extent.y)
    {
        auto grid_idx = getLinearIndex(tx, ty, extent);
        //vd_dists[grid_idx] = grid[grid_idx].z / hypotf(extent.x, extent.y);
        auto seed_idx = static_cast<int>(grid[grid_idx].w);
        auto sc = seed_colors[seed_idx];
        vd_colors[grid_idx] = {sc.x, sc.y, sc.z, 1.f};
    }
}

void jumpFlood(float *vd_dists, float4 *vd_colors, float4 *grid[], float3 *colors, int2 extent)
{
    dim3 threads(32, 32);
    dim3 blocks( (extent.x + threads.x - 1) / threads.x,
                 (extent.y + threads.y - 1) / threads.y );

    int out_idx = 0, in_idx = 1;
    for (int k = extent.x / 2; k > 0; k = k >> 1)
    {
        kernelJfa<<< blocks, threads >>>(grid[out_idx], grid[in_idx], extent, k);
        checkCuda(cudaDeviceSynchronize());
        std::swap(out_idx, in_idx);
    }
    kernelWriteResult<<< blocks, threads >>>(vd_dists, vd_colors, grid[in_idx], colors, extent);
    checkCuda(cudaDeviceSynchronize());
}

__global__
void kernelSetNonSeeds(float4 *seeds, int seed_count)
{
    auto tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx < seed_count)
    {
        seeds[tx] = make_float4(-1.f, -1.f, 0.f, -1.f);
    }
}

__global__
void kernelInitSeeds(float4 *grid, float2 *seeds, int seed_count, int2 extent)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < seed_count)
    {
        auto coord = seeds[tidx];
        int2 seed{ (int)coord.x, (int)coord.y };
        auto grid_idx = getLinearIndex(seed.x, seed.y, extent);
        grid[grid_idx] = {coord.x, coord.y, 0.f, (float)tidx};
    }
}

void initJumpFlood(float4 *d_grid, float2 *d_seeds, int seed_count, int2 extent)
{
    dim3 threads{128};
    dim3 blocks1{ (extent.x * extent.y + threads.x - 1) / threads.x};
    dim3 blocks2{ (seed_count + threads.x - 1) / threads.x};

    kernelSetNonSeeds<<< blocks1, threads >>>(d_grid, extent.x * extent.y);
    checkCuda(cudaDeviceSynchronize());
    kernelInitSeeds<<< blocks2, threads >>>(
        d_grid, d_seeds, seed_count, extent
    );
    checkCuda(cudaDeviceSynchronize());
}

__global__
void initSystem(float2 *seeds, float3 *seed_colors, size_t n,
    curandState *states, int2 extent, unsigned seed)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < n)
    {
        auto local_state = states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform(&local_state);
        auto ry = extent.y * curand_uniform(&local_state);
        float2 p{rx, ry};
        seeds[tidx] = p;

        float r = curand_uniform(&local_state);
        float g = curand_uniform(&local_state);
        float b = curand_uniform(&local_state);
        seed_colors[tidx] = {r, g, b};

        states[tidx] = local_state;
    }
}

__device__ float clamp(float x, float low, float high)
{
    return fmaxf(low, fminf(high, x));
}

__global__
void integrate2d(float2 *coords, size_t n, curandState *states, int2 extent)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < n)
    {
        auto local_state = states[tidx];
        auto r = curand_normal2(&local_state);
        auto p = coords[tidx];
        p.x = clamp(p.x + r.x, 1e-6f, extent.x);
        p.y = clamp(p.y + r.y, 1e-6f, extent.y);
        coords[tidx] = p;
        states[tidx] = local_state;
    }
}

int main(int argc, char *argv[])
{
    unsigned point_count = 100;
    size_t iter_count = 10000;
    int grid_size = 128;
    if (argc >= 2) point_count = std::stoul(argv[1]);
    if (argc >= 3) grid_size   = std::stoul(argv[2]);
    if (argc >= 4) iter_count  = std::stod(argv[3]);

    float *d_vd_dists     = nullptr;
    float4 *d_vd_colors   = nullptr;
    float2 *d_coords      = nullptr;
    float4 *d_grid[2]     = {nullptr, nullptr};
    float3 *d_colors      = nullptr;
    int2 extent           = {grid_size, grid_size};
    curandState *d_states = nullptr;

    EngineHandle engine = nullptr;
    createEngine(1920, 1080, &engine);

    AllocHandle seeds = nullptr, colors = nullptr;
    allocLinear(engine, (void**)&d_coords, sizeof(float2) * point_count, &seeds);
    allocLinear(engine, (void**)&d_vd_colors, sizeof(float4) * extent.x * extent.y, &colors);

    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc;
    desc.layout      = Layout::make(point_count);
    desc.domain_type = DomainType::Domain2D;
    desc.view_type   = ViewType::Markers;
    desc.attributes[AttributeType::Position] = {
        .source = seeds,
        .size   = point_count,
        .format = FormatDescription::make<float2>(),
    };
    desc.default_size  = 1.f;
    createView(engine, &desc, &v1);

    desc.layout    = Layout::make(extent.x, extent.y);
    desc.view_type = ViewType::Voxels;
    desc.attributes[AttributeType::Position] =
        makeStructuredGrid(engine, desc.layout, {0.f,0.f,0.4999f});
    desc.attributes[AttributeType::Color] = {
        .source = colors,
        .size   = (uint)(extent.x * extent.y),
        .format = FormatDescription::make<float4>(),
    };
    createView(engine, &desc, &v2);

    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
    checkCuda(cudaMalloc(&d_colors, sizeof(float3) * point_count));

    dim3 threads{128};
    dim3 blocks { (point_count + threads.x - 1) / threads.x};
    initSystem<<<blocks, threads>>>(d_coords, d_colors, point_count, d_states, extent, 1234);
    checkCuda(cudaDeviceSynchronize());

    // Allocate device numeric canvas
    size_t seed_sizes = sizeof(float4) * (extent.x + 1) * (extent.y + 1);
    checkCuda(cudaMalloc(&d_grid[0], seed_sizes));
    checkCuda(cudaMalloc(&d_grid[1], seed_sizes));
    checkCuda(cudaDeviceSynchronize());
    initJumpFlood(d_grid[1], d_coords, point_count, extent);

    // Start rendering loop
    auto timestep_function = [&]
    {
        dim3 threads{128};
        dim3 blocks { (point_count + threads.x - 1) / threads.x};

        integrate2d<<< blocks, threads >>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
        initJumpFlood(d_grid[1], d_coords, point_count, extent);
        jumpFlood(d_vd_dists, d_vd_colors, d_grid, d_colors, extent);
    };
    display(engine, timestep_function, iter_count);

    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(d_grid[0]));
    checkCuda(cudaFree(d_grid[1]));
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_colors));
    checkCuda(cudaFree(d_vd_colors));
    checkCuda(cudaFree(d_vd_dists));
    checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
