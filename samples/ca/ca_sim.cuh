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

// Conway's Game of Life: one step with toroidal wrapping.
void launchStepGoL(const uint8_t* src, uint8_t* dst, int W, int H, cudaStream_t s = 0);

// out[i] = 255 - grid[i]. Used only by the datoviz benchmark to counter datoviz's
// built-in DVZ_CMAP_BINARY colormap, which maps value 0 -> white and 255 -> black
// (the opposite of the alive=bright/dead=dark convention mimir's own R8 shader uses).
void launchInvertGrid(const uint8_t* grid, uint8_t* out, int W, int H, cudaStream_t s = 0);
