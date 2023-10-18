#include <cudaview/cudaview.hpp>

#include <iostream>
#include <limits> // std::numeric_limits

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cudaview/validation.hpp>
using namespace validation; // checkCuda

constexpr float max_distance = std::numeric_limits<float>::max();

struct JumpFloodProgram
{
    cudaStream_t stream    = nullptr;
    float *d_distances     = nullptr;
    float *d_coords        = nullptr;
    float4 *d_grid[2]      = {nullptr, nullptr};
    curandState *d_states  = nullptr;
    unsigned element_count = 0;
    int2 extent            = {0, 0};

    JumpFloodProgram(unsigned particle_count, int width, int height);
    void setInitialState();
    void cleanup();
    void runTimestep();
};

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
        distances[grid_idx] = seeds[grid_idx].w / hypotf(extent.x, extent.y);
    }
}

void jumpFlood(float *distances, float4 *seeds[], int2 extent, cudaStream_t stream)
{
    dim3 threads(32, 32);
    dim3 blocks( (extent.x + threads.x - 1) / threads.x,
                (extent.y + threads.y - 1) / threads.y );

    int out_idx = 0, in_idx = 1;
    for (int k = extent.x / 2; k > 0; k = k >> 1)
    {
        kernelJfa<<< blocks, threads, 0, stream >>>(
        seeds[out_idx], seeds[in_idx], extent, k
        );
        checkCuda(cudaDeviceSynchronize());
        std::swap(out_idx, in_idx);
    }
    kernelDistanceTransform<<< blocks, threads, 0, stream >>>(
        distances, seeds[in_idx], extent
    );
    checkCuda(cudaDeviceSynchronize());
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
void kernelSetSeeds(float4 *seeds, float *raw_coords,
    int coord_count, int2 extent)
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
    int2 extent, cudaStream_t stream)
{
    dim3 threads{128};
    dim3 blocks1{ (extent.x * extent.y + threads.x - 1) / threads.x};
    dim3 blocks2{ (coord_count + threads.x - 1) / threads.x};

    kernelSetNonSeeds<<< blocks1, threads, 0, stream >>>(d_seeds, extent.x * extent.y);
    checkCuda(cudaStreamSynchronize(stream));
    kernelSetSeeds<<< blocks2, threads, 0, stream >>>(
        d_seeds, d_coords, coord_count, extent
    );
    checkCuda(cudaStreamSynchronize(stream));
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

JumpFloodProgram::JumpFloodProgram(unsigned point_count, int width, int height):
    element_count{point_count}, extent{width, height}
{
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void JumpFloodProgram::setInitialState()
{
    checkCuda(cudaSetDevice(0));

    //checkCuda(cudaMalloc(&_d_distances, dist_size));
    //checkCuda(cudaMalloc(&d_coords, sizeof(float2) * element_count));
    checkCuda(cudaMallocAsync(&d_states, sizeof(curandState) * element_count, stream));

    dim3 threads{128};
    dim3 blocks { (element_count + threads.x - 1) / threads.x};
    initSystem<<<blocks, threads>>>(d_coords, element_count, d_states, extent, 1234);
    checkCuda(cudaDeviceSynchronize());

        // Allocate device numeric canvas
        size_t seed_sizes = sizeof(float4) * extent.x * extent.y;
        checkCuda(cudaMalloc(&d_grid[0], seed_sizes));
    checkCuda(cudaMalloc(&d_grid[1], seed_sizes));
    checkCuda(cudaDeviceSynchronize());
        initJumpFlood(d_grid[1], d_coords, element_count, extent, stream);
}

void JumpFloodProgram::cleanup()
{
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaStreamDestroy(stream));
    checkCuda(cudaFree(d_grid[0]));
    checkCuda(cudaFree(d_grid[1]));
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_distances));
    checkCuda(cudaFree(d_coords));
    checkCuda(cudaDeviceReset());
}

void JumpFloodProgram::runTimestep()
{
    dim3 threads{128};
    dim3 blocks { (element_count + threads.x - 1) / threads.x};

    integrate2d<<< blocks, threads, 0, stream >>>(
        d_coords, element_count, d_states, extent
    );
    checkCuda(cudaDeviceSynchronize());

    initJumpFlood(d_grid[1], d_coords, element_count, extent, stream);

    jumpFlood(d_distances, d_grid, extent, stream);
}

int main(int argc, char *argv[])
{
    unsigned point_count = 100;
    size_t iter_count = 10000;
    if (argc >= 2)
    {
        point_count = std::stoul(argv[1]);
    }
    if (argc >= 3)
    {
        iter_count = std::stoul(argv[2]);
    }

    JumpFloodProgram program(point_count, 512, 512);
    try
    {
        CudaviewEngine engine;
        engine.init(1920, 1080);

        ViewParams params;
        params.element_count = program.element_count;
        params.extent = {(unsigned)program.extent.x, (unsigned)program.extent.y, 1};
        params.data_type = DataType::Float;
        params.channel_count = 2;
        params.data_domain = DataDomain::Domain2D;
        params.resource_type = ResourceType::UnstructuredBuffer;
        params.primitive_type = PrimitiveType::Points;
        params.cuda_stream = program.stream;
        params.options.color = {0,0,1,1};
        engine.createView((void**)&program.d_coords, params);
        //cudaMalloc((void**)&program.d_coords, sizeof(float2) * point_count);

        params.element_count = program.extent.x * program.extent.y;
        params.resource_type = ResourceType::TextureLinear;
        params.data_type = DataType::Float;
        params.channel_count = 1;
        engine.createView((void**)&program.d_distances, params);
        //cudaMalloc((void**)&program.d_distances, sizeof(float) * program.extent.x * program.extent.y);

        program.setInitialState();

        // Start rendering loop
        auto timestep_function = std::bind(&JumpFloodProgram::runTimestep, program);
        engine.display(timestep_function, iter_count);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
        program.cleanup();
    }
    program.cleanup();

    return EXIT_SUCCESS;
}
