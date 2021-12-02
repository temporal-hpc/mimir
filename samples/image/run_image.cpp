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
    VulkanEngine engine(program.extent, program.stream);
    engine.init(800, 600);
    engine.registerStructuredMemory((void**)&program.d_pixels,
      program.extent.x, program.extent.y, sizeof(uchar4), DataFormat::Rgba32
    );

    program.setInitialState();
    auto timestep_function = std::bind(&ImageProgram::runTimestep, program);
    engine.registerFunction(timestep_function, iter_count);

    // Start rendering loop
    engine.display();
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
