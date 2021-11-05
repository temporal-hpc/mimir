#include "jump_flood.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
  size_t point_count = 10;
  size_t iter_count = 10000;
  if (argc >= 2)
  {
    point_count = std::stoul(argv[1]);
  }
  if (argc >= 3)
  {
    iter_count = std::stoul(argv[2]);
  }

  JumpFloodProgram program(point_count, 512, 512);
  try
  {
    VulkanCudaEngine engine(program._extent, program._stream);
    engine.init(800, 600);
    auto coord_memory = engine.registerUnstructuredMemory(
      program._element_count, sizeof(float2));
    program._d_coords = reinterpret_cast<float*>(coord_memory);
    engine.registerStructuredMemory(
      program._d_distances, program._extent.x, program._extent.y
    );

    program.setInitialState();
    program.runTimestep();//program.runTimestep();program.runTimestep();

    auto timestep_function = std::bind(&JumpFloodProgram::runTimestep, program);
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
