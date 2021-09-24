#include "cudaprogram.hpp"

#include <cstdio>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

__global__ void initSystem(float *coords, size_t particle_count,
  curandState *global_states, size_t state_count, int2 extent, unsigned seed)
{
  auto particles = reinterpret_cast<float2*>(coords);
  auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < particle_count)
  {
    auto local_state = global_states[tidx];
    curand_init(seed, tidx, 0, &local_state);
    auto rx = curand_uniform(&local_state);
    auto ry = curand_uniform(&local_state);
    float2 p{rx * extent.x, ry * extent.y};
    particles[tidx] = p;
    global_states[tidx] = local_state;
  }
}

__global__ void integrate2d(float *coords, size_t particle_count,
  curandState *global_states, size_t state_count, int2 extent)
{
  auto particles = reinterpret_cast<float2*>(coords);
  auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < particle_count)
  {
    auto local_state = global_states[tidx];
    auto r = curand_normal2(&local_state);
    auto p = particles[tidx];
    p.x += r.x;
    if (p.x > extent.x) p.x = extent.x;
    p.y += r.y;
    if (p.y > extent.y) p.y = extent.y;
    particles[tidx] = p;
    global_states[tidx] = local_state;
  }
}

void CudaProgram::init(size_t particle_count, int width, int height, unsigned seed)
{
  _particle_count = particle_count;
  _state_count = _particle_count;
  _bounding_box = {width, height};
  _grid_size = (_particle_count + _block_size - 1) / _block_size;

  checkCuda(cudaMalloc(&_d_coords, sizeof(float2) * _particle_count));
  checkCuda(cudaMalloc(&_d_states, sizeof(curandState) * _state_count));
  initSystem<<<_grid_size, _block_size>>>(
    _d_coords, _particle_count, _d_states, _state_count, _bounding_box, seed
  );
  checkCuda(cudaDeviceSynchronize());
}

void CudaProgram::cleanup()
{
  checkCuda(cudaFree(_d_coords));
  checkCuda(cudaFree(_d_states));
}

void CudaProgram::runTimestep()
{
  integrate2d<<<_grid_size, _block_size>>>(
    _d_coords, _particle_count, _d_states, _state_count, _bounding_box
  );
  checkCuda(cudaDeviceSynchronize());
}
