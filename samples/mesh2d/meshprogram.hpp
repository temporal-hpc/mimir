#pragma once

#include <cuda_runtime.h>

struct MeshProgram
{
  float2 *d_points    = nullptr;
  int3 *d_triangles   = nullptr;
  int2 extent         = {1, 1};

  MeshProgram();
  void setInitialState();
  void cleanup();
  void runTimestep();
};
