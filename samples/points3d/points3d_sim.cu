#include "points3d_sim.cuh"
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Displacement stddev per axis per step, in world units (cube spans [-1,1]).
constexpr float STEP_SIGMA = 0.02f;

__global__ void initRngKernel(curandState* states, int state_count, uint32_t seed)
{
    int tidx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tidx >= state_count) return;
    curand_init(seed, tidx, 0, &states[tidx]);
}

__global__ void initPosKernel(float* coords, size_t point_count, curandState* rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state  = rng[tidx];
    for (size_t i = tidx; i < point_count; i += stride)
    {
        // curand_uniform is in (0,1]; map to [-1,1].
        points[i] = {
            2.f * curand_uniform(&state) - 1.f,
            2.f * curand_uniform(&state) - 1.f,
            2.f * curand_uniform(&state) - 1.f,
        };
    }
    rng[tidx] = state;
}

__global__ void integrate3dKernel(float* coords, size_t point_count, curandState* rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state  = rng[tidx];
    for (size_t i = tidx; i < point_count; i += stride)
    {
        auto p = points[i];
        p.x = fminf(fmaxf(p.x + STEP_SIGMA * curand_normal(&state), -1.f), 1.f);
        p.y = fminf(fmaxf(p.y + STEP_SIGMA * curand_normal(&state), -1.f), 1.f);
        p.z = fminf(fmaxf(p.z + STEP_SIGMA * curand_normal(&state), -1.f), 1.f);
        points[i] = p;
    }
    rng[tidx] = state;
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

RngStates createRngStates(uint32_t seed)
{
    int device_id = -1;
    cudaGetDevice(&device_id);

    int sm_count = -1, max_sm_threads = -1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    cudaDeviceGetAttribute(&max_sm_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);

    RngStates rng;
    rng.block = 256;
    rng.count = (max_sm_threads / rng.block) * sm_count * rng.block;
    rng.grid  = (rng.count + rng.block - 1) / rng.block;

    curandState* states = nullptr;
    cudaMalloc(&states, sizeof(curandState) * rng.count);
    rng.states = states;

    initRngKernel<<<rng.grid, rng.block>>>(states, rng.count, seed);
    cudaDeviceSynchronize();
    return rng;
}

void destroyRngStates(RngStates& rng)
{
    cudaFree(rng.states);
    rng.states = nullptr;
}

size_t rngStatesBytes(const RngStates& rng)
{
    return sizeof(curandState) * (size_t)rng.count;
}

void launchInitPositions(float* pos3, size_t point_count, RngStates& rng)
{
    initPosKernel<<<rng.grid, rng.block>>>(pos3, point_count, (curandState*)rng.states);
    cudaDeviceSynchronize();
}

void launchIntegrate3D(float* pos3, size_t point_count, RngStates& rng, cudaStream_t s)
{
    integrate3dKernel<<<rng.grid, rng.block, 0, s>>>(pos3, point_count, (curandState*)rng.states);
}
