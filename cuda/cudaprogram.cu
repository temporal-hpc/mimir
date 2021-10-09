#include "cudaprogram.hpp"

#include <cstdio>

#include <experimental/source_location>

using source_location = std::experimental::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
  source_location src = source_location::current())
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA assertion: %s on function %s at %s(%d)\n",
      cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
    );
    if (panic) exit(code);
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
    auto rx = 2.f * curand_uniform(&local_state) - 1.f;
    auto ry = 2.f * curand_uniform(&local_state) - 1.f;
    float2 p{rx, ry};
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
    p.x += r.x / (4*extent.x);
    if (p.x > 1.f) p.x = 1.f;
    if (p.x < -1.f) p.x = -1.f;
    p.y += r.y / (4*extent.y);
    if (p.y > 1.f) p.y = 1.f;
    if (p.y < -1.f) p.y = -1.f;
    particles[tidx] = p;
    global_states[tidx] = local_state;
  }
}

CudaProgram::CudaProgram(size_t particle_count, int width, int height, unsigned seed):
  _particle_count(particle_count), _bounding_box{width, height},
  _state_count(_particle_count), _seed(seed),
  _grid_size((_particle_count + _block_size - 1) / _block_size)
{
  checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

void CudaProgram::setInitialState()
{
  //checkCuda(cudaMalloc(&_d_coords, sizeof(float2) * _particle_count));
  checkCuda(cudaMalloc(&_d_states, sizeof(curandState) * _state_count));
  initSystem<<<_grid_size, _block_size>>>(
    _d_coords, _particle_count, _d_states, _state_count, _bounding_box, _seed
  );
  //checkCuda(cudaDeviceSynchronize());
}

void CudaProgram::registerBuffer(float *d_buffer)
{
  _d_coords = d_buffer;
}

void CudaProgram::cleanup()
{
  checkCuda(cudaStreamDestroy(_stream));
  checkCuda(cudaFree(_d_states));
}

void CudaProgram::runTimestep()
{
  integrate2d<<< _grid_size, _block_size, 0, _stream >>>(
    _d_coords, _particle_count, _d_states, _state_count, _bounding_box
  );
  //checkCuda(cudaStreamSynchronize(stream));
}
