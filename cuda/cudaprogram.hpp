#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct CudaProgram
{
  float *_d_coords = nullptr;
  size_t _particle_count = 0;
  curandState *_d_states = nullptr;
  size_t _state_count = 0;
  int2 _bounding_box{0, 0};
  unsigned _block_size = 256;
  unsigned _grid_size = 0;

  void init(size_t particle_count, int width, int height, unsigned seed = 0);
  void cleanup();
  void runTimestep();
};
