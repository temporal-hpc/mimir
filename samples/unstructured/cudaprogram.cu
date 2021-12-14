#include "cudaprogram.hpp"
#include "cuda_utils.hpp"

__global__ void initSystem(float *coords, size_t particle_count,
  curandState *global_states, size_t state_count, int2 extent, unsigned seed)
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
  curandState *global_states, size_t state_count, int2 extent)
{
  auto particles = reinterpret_cast<float2*>(coords);
  auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < particle_count)
  {
    auto local_state = global_states[tidx];
    auto r = curand_normal2(&local_state);
    auto p = particles[tidx];
    p.x += r.x / 5.f;
    if (p.x > extent.x) p.x = extent.x;
    if (p.x < 0) p.x = 0;
    p.y += r.y / 5.f;
    if (p.y > extent.y) p.y = extent.y;
    if (p.y < 0) p.y = 0;
    particles[tidx] = p;
    global_states[tidx] = local_state;
  }
}

CudaProgram::CudaProgram(size_t particle_count, int width, int height, unsigned seed):
  particle_count(particle_count), bounding_box{width, height},
  state_count(particle_count), seed(seed),
  grid_size((particle_count + block_size - 1) / block_size)
{
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void CudaProgram::setInitialState()
{
  //checkCuda(cudaMalloc(&d_coords, sizeof(float2) * particle_count));
  checkCuda(cudaMalloc(&d_states, sizeof(curandState) * state_count));
  initSystem<<<grid_size, block_size>>>(
    d_coords, particle_count, d_states, state_count, bounding_box, seed
  );
  //checkCuda(cudaDeviceSynchronize());
}

void CudaProgram::cleanup()
{
  checkCuda(cudaStreamDestroy(stream));
  checkCuda(cudaFree(d_states));
  checkCuda(cudaFree(d_coords));
}

void CudaProgram::runTimestep()
{
  integrate2d<<< grid_size, block_size, 0, stream >>>(
    d_coords, particle_count, d_states, state_count, bounding_box
  );
  //checkCuda(cudaStreamSynchronize(stream));
}
