#include "image.hpp"
#include "cudaview/vk_engine.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
  size_t point_count = 10;
  size_t iter_count = 10000;
  if (argc >= 2)
  {
    point_count = std::stoul(argv[1]);
  }

  ImageProgram program;
  try
  {
    int3 extent{program.extent.x, program.extent.y, 1};
    VulkanEngine engine(extent, program.stream);
    engine.init(800, 600);
    engine.registerStructuredMemory((void**)&program.d_pixels,
      program.extent.x, program.extent.y, sizeof(uchar4), DataFormat::Rgba32
    );
    program.setInitialState();

    // Start rendering loop
    auto timestep_function = std::bind(&ImageProgram::runTimestep, program);
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
