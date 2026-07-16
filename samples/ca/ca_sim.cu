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

    // Linear cell indices must be 64-bit: for large grids (e.g. 50000 x 50000) row*W exceeds
    // INT_MAX (~2.15e9), so computing the index in int overflows to a negative/wrapped offset
    // and the kernel reads/writes out of bounds -> "illegal memory access".
    const size_t Wz = (size_t)W;
    int n = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            size_t nidx = (size_t)((y + dy + H) % H) * Wz + (size_t)((x + dx + W) % W);
            n += (src[nidx] != 0) ? 1 : 0;
        }

    size_t idx = (size_t)y * Wz + (size_t)x;
    bool alive = src[idx] != 0;
    dst[idx] = (n == 3 || (alive && n == 2)) ? 255u : 0u;
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

// Per-cell hash -> uniform [0,1). Deterministic from (seed,x,y); no per-cell RNG state.
__device__ inline float cellRand01(uint32_t seed, uint32_t x, uint32_t y)
{
    uint32_t h = seed + 0x9e3779b9u;
    h ^= x * 0x85ebca6bu; h *= 0xc2b2ae35u; h ^= h >> 15;
    h ^= y * 0x27d4eb2fu; h *= 0x165667b1u; h ^= h >> 13;
    return (h >> 8) * (1.0f / 16777216.0f);  // top 24 bits -> [0,1)
}

__global__ void initGridKernel(uint8_t* g, int W, int H, uint32_t seed, float density)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;
    g[(size_t)y * (size_t)W + (size_t)x] =
        (cellRand01(seed, (uint32_t)x, (uint32_t)y) < density) ? 255u : 0u;
}

void launchInitGrid(uint8_t* g, int W, int H, uint32_t seed, float density, cudaStream_t s)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    initGridKernel<<<grid, block, 0, s>>>(g, W, H, seed, density);
}

// Output texel (dx,dy) maps to the source cell block [x0,x1) x [y0,y1), where the (vw x vh) cell
// window at (ox,oy) is spread over the DW x DH texels. Stores the live-cell FRACTION (0..255) of
// that block: a coverage average, so minified views show density instead of saturating to white.
// When vw < DW the block collapses to a single cell (guard x1 = x0+1), giving crisp nearest-neighbour
// magnification. Indices are 64-bit: row*W overflows int for large grids.
__global__ void resampleKernel(const uint8_t* __restrict__ src,
                               uint8_t*       __restrict__ dst,
                               int W, int H, int DW, int DH, int ox, int oy, int vw, int vh)
{
    int dx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int dy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (dx >= DW || dy >= DH) return;

    const size_t Wz = (size_t)W;
    long x0 = ox + (long)dx * vw / DW, x1 = ox + (long)(dx + 1) * vw / DW;
    long y0 = oy + (long)dy * vh / DH, y1 = oy + (long)(dy + 1) * vh / DH;
    if (x1 <= x0) x1 = x0 + 1;   // magnify: at least one source cell per texel
    if (y1 <= y0) y1 = y0 + 1;
    x0 = max(x0, 0L); x1 = min(x1, (long)W);
    y0 = max(y0, 0L); y1 = min(y1, (long)H);

    uint32_t live = 0, area = 0;
    for (long y = y0; y < y1; ++y)
        for (long x = x0; x < x1; ++x)
        {
            live += (src[(size_t)y * Wz + (size_t)x] != 0) ? 1u : 0u;
            ++area;
        }

    dst[(size_t)dy * (size_t)DW + (size_t)dx] =
        area ? (uint8_t)((255u * live) / area) : 0u;
}

void launchResample(const uint8_t* src, uint8_t* dst,
                    int W, int H, int DW, int DH,
                    int ox, int oy, int vw, int vh, cudaStream_t s)
{
    dim3 block(16, 16);
    dim3 grid((DW + 15) / 16, (DH + 15) / 16);
    resampleKernel<<<grid, block, 0, s>>>(src, dst, W, H, DW, DH, ox, oy, vw, vh);
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
