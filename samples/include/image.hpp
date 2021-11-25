#include <cuda_runtime.h>

struct ImageProgram
{
  cudaStream_t stream = nullptr;
  uchar4 *d_pixels    = nullptr;
  int2 extent         = {512, 512};

  ImageProgram();
  void setInitialState();
  void cleanup();
  void runTimestep();
};
