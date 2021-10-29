#include "jump_flood.hpp"
#include "image.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
  size_t point_count = 10;
  size_t iter_count = 1000;
  if (argc >= 2)
  {
    point_count = std::stoul(argv[1]);
  }

  JumpFloodProgram program(100, 256, 256);
  //ImageProgram program;
  try
  {
    VulkanCudaEngine engine(program._extent, program._stream);
    engine.init(800, 600);
    engine.registerUnstructuredMemory(program._d_coords, program._element_count);
    engine.registerStructuredMemory(
      program._d_distances, program._extent.x, program._extent.y
    );
    /*engine.registerStructuredMemory(
      program._d_image, program._extent.x, program._extent.y
    );*/

    program.setInitialState();
    auto timestep_function = std::bind(&JumpFloodProgram::runTimestep, program);
    //auto timestep_function = std::bind(&ImageProgram::runTimestep, program);
    engine.registerFunction(timestep_function, iter_count);

    // Start rendering loop
    engine.mainLoop();
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
