#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct CAParams {
    int      width   = 1024;
    int      height  = 1024;
    uint32_t seed    = 12345;
    float    density = 0.3f;
};

// Generate initial grid on the host — deterministic from p.seed.
std::vector<uint8_t> initGrid(const CAParams& p);

// Generate the initial grid directly on the GPU: each cell is set alive from a per-cell hash of
// (seed, x, y) with probability `density`. Deterministic from the seed, and avoids the multi-GB
// host RNG fill + H2D copy that stalls startup for seconds on large grids.
void launchInitGrid(uint8_t* g, int W, int H, uint32_t seed, float density, cudaStream_t s = 0);

// Conway's Game of Life: one step with toroidal wrapping.
void launchStepGoL(const uint8_t* src, uint8_t* dst, int W, int H, cudaStream_t s = 0);

// Resample a W x H grid into a DW x DH display buffer, showing the cell window
//   [ox, ox+vw) x [oy, oy+vh)
// mapped onto the DW x DH texels. Output stores the LIVE-CELL FRACTION (0..255) of each texel's
// source block -- a coverage average, so a coarse (minified) view shows density (grey) instead of
// saturating to white the way a max-pool would. When the window is SMALLER than the display
// (vw < DW), each cell maps to several texels (nearest-neighbour magnification) and cells stay
// crisp 0/255. (ox,oy) pans; (vw,vh) zooms -- shrink to magnify, grow to fit more of the grid.
void launchResample(const uint8_t* src, uint8_t* dst,
                    int W, int H, int DW, int DH,
                    int ox, int oy, int vw, int vh, cudaStream_t s = 0);
