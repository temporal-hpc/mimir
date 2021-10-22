#include <cuda_runtime.h>

struct JumpFloodProgram
{
  cudaStream_t _stream = nullptr;
  float2 *_d_grid = nullptr;
  float *_d_distances = nullptr;
  size_t _element_count = 0;
  int2 _extent{0, 0};
  unsigned _block_size = 32;
  unsigned _grid_size = 0;

  JumpFloodProgram(size_t particle_count, int width, int height);
  void setInitialState();
  void cleanup();
  void runTimestep();
};
