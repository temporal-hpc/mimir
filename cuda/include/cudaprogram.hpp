#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct CudaProgram
{
  cudaStream_t _stream = nullptr;
  float *_d_coords = nullptr;
  size_t _particle_count = 0;
  curandState *_d_states = nullptr;
  int2 _bounding_box{0, 0};
  unsigned _block_size = 256;
  unsigned _grid_size = 0;
  size_t _state_count = 0;
  unsigned _seed = 0;

  CudaProgram(size_t particle_count, int width, int height, unsigned seed = 0);
  void setInitialState();
  void cleanup();
  void runTimestep();
};
