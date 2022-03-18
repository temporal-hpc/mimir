#include "jump_flood.hpp"
#include "cudaview/vk_engine.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
  unsigned point_count = 100;
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
    int3 extent{program.extent.x, program.extent.y, 1};
    VulkanEngine engine(extent, program.stream);
    engine.init(800, 600);
    engine.registerUnstructuredMemory((void**)&program.d_coords,
      program.element_count, sizeof(float2), UnstructuredDataType::Points,
      DataDomain::Domain2D
    );
    engine.registerStructuredMemory((void**)&program.d_distances,
      {(unsigned)program.extent.x, (unsigned)program.extent.y, 1}, sizeof(float),
      DataFormat::Float32
    );

    program.setInitialState();

    // Start rendering loop
    auto timestep_function = std::bind(&JumpFloodProgram::runTimestep, program);
    engine.display(timestep_function, iter_count);
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
