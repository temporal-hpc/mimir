#include "jump_flood.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
  size_t point_count = 10;
  if (argc >= 2)
  {
    point_count = std::stoul(argv[1]);
  }

  JumpFloodProgram program(100, 256, 256);
  try
  {
    program.setInitialState();
    program.runTimestep();
    // Initialize engine
    //VulkanCudaEngine engine(program._particle_count, program._stream);
    //engine.init(800, 600);
    //engine.registerDeviceMemory(program._d_coords);

    // Cannot make CUDA calls that use the target device memory before
    // registering it on the engine
    //program.setInitialState();

    // Set up the function that we want to display
    //auto timestep_function = std::bind(&CudaProgram::runTimestep, program);
    //engine.registerFunction(timestep_function, iter_count);

    // Start rendering loop
    //engine.mainLoop();

  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
    program.cleanup();
  }
  program.cleanup();

  return EXIT_SUCCESS;
}

//VulkanEngine engine(vertices.size());
