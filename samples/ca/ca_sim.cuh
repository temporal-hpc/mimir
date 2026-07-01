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
