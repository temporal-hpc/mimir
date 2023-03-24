#include "cudaview/vk_engine.hpp"

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include <iostream> // std::cerr
#include <string> // std::stoul

#include "cuda_utils.hpp" // checkCuda

struct CudaProgram
{
  cudaStream_t stream   = nullptr;
  float *d_coords       = nullptr;
  size_t particle_count = 0;
  curandState *d_states = nullptr;
  int2 bounding_box     = {0, 0};
  unsigned block_size   = 256;
  unsigned grid_size    = 0;
  size_t state_count    = 0;
  unsigned seed         = 0;

  CudaProgram(size_t particle_count, int width, int height, unsigned seed = 0);
  void setInitialState();
  void cleanup();
  void runTimestep();
};

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

int main(int argc, char *argv[])
{
  size_t particle_count = 100;
  size_t iter_count = 10000;
  if (argc >= 2)
  {
    particle_count = std::stoul(argv[1]);
  }
  if (argc >= 3)
  {
    iter_count = std::stoul(argv[2]);
  }

  CudaProgram program(particle_count, 200, 200, 123456);
  try
  {
    // Initialize engine
    VulkanEngine engine;
    engine.init(800, 600);
    ViewParams params;
    params.element_count = program.particle_count;
    params.element_size = sizeof(float2);
    params.extent = {200, 200, 1};
    params.data_domain = DataDomain::Domain2D;
    params.resource_type = ResourceType::UnstructuredBuffer;
    params.primitive_type = PrimitiveType::Points;
    params.cuda_stream = program.stream;
    params.options.external_shaders = {
      {"shaders/precompiled/marker_vertex2dMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
      {"shaders/precompiled/marker_geometryMain.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
      {"shaders/precompiled/marker_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };
    engine.addView((void**)&program.d_coords, params);

    // Cannot make CUDA calls that use the target device memory before
    // registering it on the engine
    program.setInitialState();

    // Set up the function that we want to display
    auto timestep_function = std::bind(&CudaProgram::runTimestep, program);
    // Start rendering loop
    engine.display(timestep_function, iter_count);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    program.cleanup();
    return EXIT_FAILURE;
  }
  program.cleanup();

  return EXIT_SUCCESS;
}

//VulkanEngine engine(vertices.size());
