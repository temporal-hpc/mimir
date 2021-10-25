#include <cuda_runtime.h>

struct ImageProgram
{
  cudaStream_t _stream = nullptr;
  unsigned char *_d_image = nullptr;
  int2 _extent{512, 512};

  ImageProgram();
  void setInitialState();
  void cleanup();
  void runTimestep();
};
