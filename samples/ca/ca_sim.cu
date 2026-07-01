#include "ca_sim.cuh"
#include <random>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Conway's Game of Life — toroidal boundary.
__global__ void stepGoLKernel(const uint8_t* __restrict__ src,
                               uint8_t*       __restrict__ dst,
                               int W, int H)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int n = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            n += (src[((y + dy + H) % H) * W + ((x + dx + W) % W)] != 0) ? 1 : 0;
        }

    bool alive = src[y * W + x] != 0;
    dst[y * W + x] = (n == 3 || (alive && n == 2)) ? 255u : 0u;
}

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------

void launchStepGoL(const uint8_t* src, uint8_t* dst, int W, int H, cudaStream_t s)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    stepGoLKernel<<<grid, block, 0, s>>>(src, dst, W, H);
}

// ---------------------------------------------------------------------------
// Host init
// ---------------------------------------------------------------------------

std::vector<uint8_t> initGrid(const CAParams& p)
{
    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<uint8_t> h((size_t)p.width * p.height);
    for (auto& c : h)
        c = dist(rng) < p.density ? 255u : 0u;
    return h;
}
