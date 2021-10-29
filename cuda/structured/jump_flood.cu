#include "jump_flood.hpp"
#include "cuda_utils.hpp"

#include <chrono>
#include <limits> // std::numeric_limits
#include <random>

constexpr float max_distance = std::numeric_limits<float>::max();

__device__ float clamp(float x, float low, float high)
{
  return fmaxf(low, fminf(high, x));
}

__global__
void kernelJumpFlood(float *distances, float2 *seeds, const int2 extent)
{
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;
  const int ty = threadIdx.y + blockIdx.y * blockDim.y;
  // TODO: Handle boundaries
  if (tx < extent.x && ty < extent.y)
  {
    auto self_idx = ty * extent.x + tx;

    auto max_extent = max(extent.x, extent.y);
    for (int step = max_extent / 2; step > 0; step = step >> 1)
    {
      auto own_seed = seeds[self_idx];
      float2 best_coord{own_seed.x, own_seed.y};
      auto best_dist = max_distance;
      for (int y = -1; y <= 1; ++y)
      {
        for (int x = -1; x <= 1; ++x)
        {
          auto lookup_x = tx + x * step;
          auto lookup_y = ty + y * step;
          if (lookup_x < 0 || lookup_x > extent.x || lookup_y < 0 || lookup_y > extent.y)
          {
            continue;
          }
          auto seed = seeds[extent.y * lookup_y + lookup_x];
          auto dist = hypotf(seed.x - tx, seed.y - ty);

          if ((seed.x != 0.f || seed.y != 0.f) && dist < best_dist)
          {
            best_dist = dist;
            best_coord = make_float2(seed.x, seed.y);
          }
        }
      }
      __syncthreads();
      distances[self_idx] = best_dist;
      seeds[self_idx] = best_coord;
    }
    auto extent_distance = hypotf(extent.x, extent.y);
    distances[self_idx] /= extent_distance;
  }
}

__global__
void kernelCreateJfaInput(float2 *seeds, float *raw_coords,
  int coord_count, int2 extent)
{
  int tx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tx < coord_count)
  {
    auto coord = reinterpret_cast<float2*>(raw_coords)[tx];
    int2 point{ (int)coord.x, (int)coord.y };
    if (point.x > 0 && point.x < extent.x && point.y > 0 && point.y < extent.y)
    {
      seeds[extent.x * point.y + point.x] = coord;
    }
  }
}

__global__ void initSystem(float *coords, size_t particle_count,
  curandState *global_states, int2 extent, unsigned seed)
{
  auto particles = reinterpret_cast<float2*>(coords);
  auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < particle_count)
  {
    auto local_state = global_states[tidx];
    curand_init(seed, tidx, 0, &local_state);
    auto rx = extent.x * curand_uniform(&local_state);
    auto ry = extent.y * curand_uniform(&local_state);
    float2 p{rx, ry};
    particles[tidx] = p;
    global_states[tidx] = local_state;
  }
}

__global__ void integrate2d(float *coords, size_t particle_count,
  curandState *global_states, int2 extent)
{
  auto particles = reinterpret_cast<float2*>(coords);
  auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < particle_count)
  {
    auto local_state = global_states[tidx];
    auto r = curand_normal2(&local_state);
    auto p = particles[tidx];
    p.x = clamp(p.x + r.x / 5.f, 0.f, extent.x);
    p.y = clamp(p.y + r.y / 5.f, 0.f, extent.y);
    particles[tidx] = p;
    global_states[tidx] = local_state;
  }
}

JumpFloodProgram::JumpFloodProgram(size_t point_count, int width, int height):
  _element_count{point_count}, _extent{width, height},
  _grid_size((_element_count + _block_size - 1) / _block_size)
{
  checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

void JumpFloodProgram::setInitialState()
{
  checkCuda(cudaMalloc(&_d_states, sizeof(curandState) * _element_count));
  initSystem<<<_grid_size, _block_size>>>(
    _d_coords, _element_count, _d_states, _extent, 1234
  );

	checkCuda(cudaSetDevice(0));

	// Allocate device numeric canvas
	size_t numericCanvasSize = sizeof(float2) * _extent.x * _extent.y;
	checkCuda(cudaMalloc((void**)&_d_grid, numericCanvasSize));
	checkCuda(cudaMemset(_d_grid, 0, numericCanvasSize));

  auto dist_size = sizeof(float) * _extent.x * _extent.y;
  //checkCuda(cudaMalloc(&_d_distances, dist_size));
  checkCuda(cudaMemset(_d_distances, 0, dist_size));
}

void JumpFloodProgram::cleanup()
{
  checkCuda(cudaStreamSynchronize(_stream));
  checkCuda(cudaStreamDestroy(_stream));
  checkCuda(cudaFree(_d_grid));
  checkCuda(cudaFree(_d_states));
  //checkCuda(cudaFree(_d_distances));
  //checkCuda(cudaFree(_d_coords));
  checkCuda(cudaDeviceReset());
}

void JumpFloodProgram::runTimestep()
{
  integrate2d<<< _grid_size, _block_size, 0, _stream >>>(
    _d_coords, _element_count, _d_states, _extent
  );
  checkCuda(cudaStreamSynchronize(_stream));

  checkCuda(cudaMemset(_d_grid, 0, sizeof(float2) * _extent.x * _extent.y));
  kernelCreateJfaInput<<< _grid_size, _block_size, 0, _stream >>>(
    _d_grid, _d_coords, _element_count, _extent
  );

  dim3 block(32, 32);
  dim3 grid{ (_extent.x + block.x - 1) / block.x,
             (_extent.y + block.y - 1) / block.y };
  kernelJumpFlood<<< grid, block, 0, _stream >>>(_d_distances, _d_grid, _extent);
  checkCuda(cudaStreamSynchronize(_stream));
}
