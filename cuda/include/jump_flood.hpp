#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct JumpFloodProgram
{
  cudaStream_t stream    = nullptr;

  float *d_distances     = nullptr;
  float *d_coords        = nullptr;

  float4 *d_grid[2]      = {nullptr, nullptr};
  curandState *d_states  = nullptr;

  unsigned element_count = 0;
  int2 extent            = {0, 0};

  JumpFloodProgram(size_t particle_count, int width, int height);
  void setInitialState();
  void cleanup();
  void runTimestep();
};
