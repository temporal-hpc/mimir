#include "kmodal_sim.cuh"
#include <curand_kernel.h>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Displacement stddev per axis per step, in world units (cube spans [-1,1]). Small = gentle,
// slow per-iteration motion; the reversion rate lambda is derived from this, so the equilibrium
// blob size (epsilon) is unaffected -- the clusters just evolve more slowly.
constexpr float STEP_SIGMA = 0.002f;

__global__ void initRngKernel(curandState* states, int state_count, uint32_t seed)
{
    int tidx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tidx >= state_count) return;
    curand_init(seed, tidx, 0, &states[tidx]);
}

__global__ void initPosKernel(float* coords, size_t point_count, const float3* centers,
                              unsigned int* ids, unsigned int k, float epsilon,
                              curandState* rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state  = rng[tidx];
    for (size_t i = tidx; i < point_count; i += stride)
    {
        // curand_uniform is in (0,1], so u*k can reach k exactly; clamp to k-1.
        auto c = min((unsigned int)(curand_uniform(&state) * k), k - 1);
        ids[i] = c;
        auto ctr = centers[c];
        points[i] = {
            fminf(fmaxf(ctr.x + epsilon * curand_normal(&state), -1.f), 1.f),
            fminf(fmaxf(ctr.y + epsilon * curand_normal(&state), -1.f), 1.f),
            fminf(fmaxf(ctr.z + epsilon * curand_normal(&state), -1.f), 1.f),
        };
    }
    rng[tidx] = state;
}

__global__ void integrate3dKernel(float* coords, size_t point_count,
                                  const float3* centers, const unsigned int* ids,
                                  float lambda, curandState* rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state  = rng[tidx];
    for (size_t i = tidx; i < point_count; i += stride)
    {
        auto p   = points[i];
        auto ctr = centers[ids[i]];
        p.x += STEP_SIGMA * curand_normal(&state) - lambda * (p.x - ctr.x);
        p.y += STEP_SIGMA * curand_normal(&state) - lambda * (p.y - ctr.y);
        p.z += STEP_SIGMA * curand_normal(&state) - lambda * (p.z - ctr.z);
        p.x = fminf(fmaxf(p.x, -1.f), 1.f);
        p.y = fminf(fmaxf(p.y, -1.f), 1.f);
        p.z = fminf(fmaxf(p.z, -1.f), 1.f);
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

ClusterData createClusters(const PointsParams& params)
{
    ClusterData clusters;
    clusters.k = params.k > 0 ? params.k : 1;

    // OU reversion rate: the discrete walk p += sigma*N(0,1) - lambda*(p - ctr) has
    // stationary per-axis stddev sigma/sqrt(2*lambda - lambda^2); solving for stddev
    // == epsilon gives lambda ~= sigma^2 / (2*epsilon^2) (small-lambda limit).
    // Capped at 1 (lambda == 1 snaps straight to the center each step).
    if (params.epsilon > 0.f)
        clusters.lambda = fminf(
            (STEP_SIGMA * STEP_SIGMA) / (2.f * params.epsilon * params.epsilon), 1.f);
    else
        clusters.lambda = 1.f;

    // Cluster centers from a host-side RNG (identical in both benchmarks), drawn
    // over the FULL cube on purpose: a center that lands near a wall gets its blob
    // sliced flat by the [-1,1] clamp, which makes the bounding cube visible in the
    // render (flat cut faces on otherwise round blobs).
    std::mt19937 gen(params.seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float3> centers(clusters.k);
    for (auto& c : centers) { c.x = dist(gen); c.y = dist(gen); c.z = dist(gen); }

    float3* d_centers = nullptr;
    cudaMalloc(&d_centers, sizeof(float3) * clusters.k);
    cudaMemcpy(d_centers, centers.data(), sizeof(float3) * clusters.k,
        cudaMemcpyHostToDevice);
    clusters.centers = d_centers;

    unsigned int* d_ids = nullptr;
    cudaMalloc(&d_ids, sizeof(unsigned int) * params.count);
    clusters.ids = d_ids;

    return clusters;
}

void destroyClusters(ClusterData& clusters)
{
    cudaFree(clusters.centers);
    cudaFree(clusters.ids);
    clusters.centers = nullptr;
    clusters.ids     = nullptr;
}

size_t clusterBytes(const ClusterData& clusters, size_t point_count)
{
    return sizeof(float3) * clusters.k + sizeof(unsigned int) * point_count;
}

void launchInitPositions(float* pos3, const PointsParams& params,
                         ClusterData& clusters, RngStates& rng)
{
    initPosKernel<<<rng.grid, rng.block>>>(
        pos3, params.count, (const float3*)clusters.centers,
        (unsigned int*)clusters.ids, clusters.k, params.epsilon,
        (curandState*)rng.states);
    cudaDeviceSynchronize();
}

void launchIntegrate3D(float* pos3, size_t point_count, const ClusterData& clusters,
                       RngStates& rng, cudaStream_t s)
{
    integrate3dKernel<<<rng.grid, rng.block, 0, s>>>(
        pos3, point_count, (const float3*)clusters.centers,
        (const unsigned int*)clusters.ids, clusters.lambda, (curandState*)rng.states);
}
