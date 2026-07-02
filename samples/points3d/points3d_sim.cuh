#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// Random-walk 3D point cloud, shared by benchmark_mimir and benchmark_datoviz.
// Points live in the [-1,1]^3 cube (centered at the origin so both renderers show
// the same centered cloud); each step adds a gaussian displacement per axis and
// clamps back to the cube.

struct PointsParams {
    unsigned int count = 1'000'000;
    uint32_t     seed  = 12345;
};

// Persistent curand states, sized to fill the device (one state per resident thread).
// `states` is a curandState* kept opaque so this header stays includable from .cpp TUs.
struct RngStates {
    void* states = nullptr;
    int   grid   = 0;
    int   block  = 0;
    int   count  = 0;
};

RngStates createRngStates(uint32_t seed);
void destroyRngStates(RngStates& rng);
size_t rngStatesBytes(const RngStates& rng);

// Uniform initial positions in [-1,1]^3 — deterministic from the seed.
void launchInitPositions(float* pos3, size_t point_count, RngStates& rng);

// One random-walk step: per-axis gaussian displacement, clamped to the cube.
void launchIntegrate3D(float* pos3, size_t point_count, RngStates& rng, cudaStream_t s = 0);
