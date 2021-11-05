#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct CudaProgram
{
  cudaStream_t stream   = nullptr;
  float *d_coords       = nullptr;
  size_t particle_count = 0;
  curandState *d_states = nullptr;
  int2 bounding_box     = {0, 0};
  unsigned block_size   = 256;
  unsigned grid_size    = 0;
  size_t state_count    = 0;
  unsigned seed         = 0;

  CudaProgram(size_t particle_count, int width, int height, unsigned seed = 0);
  void setInitialState();
  void cleanup();
  void runTimestep();
};
