#include "cudaprogram.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <iostream> // std::cerr
#include <string> // std::stoul

int main(int argc, char *argv[])
{
  using namespace std::placeholders;
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
  //VulkanEngine engine(vertices.size());
  try
  {
    VulkanCudaEngine engine(program._particle_count);
    // Initialize engine
    engine.init(800, 600);
    auto d_memory = engine.getDeviceMemory();
    program.registerBuffer(d_memory);
    program.setInitialState();

    // Set up the function that we want to display
    auto timestep_function = std::bind(&CudaProgram::runTimestep, program, _1);
    engine.registerFunction(timestep_function, iter_count);

    // Start rendering loop
    engine.mainLoop();
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
