#include "cudaprogram.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <iostream> // std::cerr
#include <string> // std::stoul

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
    VulkanCudaEngine engine({200, 200}, program.stream);
    engine.init(800, 600);
    engine.registerUnstructuredMemory(
      (void**)&program.d_coords, program.particle_count, sizeof(float2)
    );

    // Cannot make CUDA calls that use the target device memory before
    // registering it on the engine
    program.setInitialState();

    // Set up the function that we want to display
    auto timestep_function = std::bind(&CudaProgram::runTimestep, program);
    engine.registerFunction(timestep_function, iter_count);

    // Start rendering loop
    engine.display();
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
