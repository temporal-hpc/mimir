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
    VulkanEngine engine(program.stream);
    engine.init(800, 600);

    ViewParams params;
    params.element_count = program.element_count;
    params.element_size = sizeof(float2);
    params.extent = {(unsigned)program.extent.x, (unsigned)program.extent.y, 1};
    params.data_domain = DataDomain::Domain2D;
    params.resource_type = ResourceType::UnstructuredBuffer;
    params.primitive_type = PrimitiveType::Points;
    engine.addView((void**)&program.d_coords, params);

    params.element_count = program.extent.x * program.extent.y;
    params.element_size = sizeof(float);
    params.resource_type = ResourceType::TextureLinear;
    params.texture_format = TextureFormat::Float32;
    engine.addView((void**)&program.d_distances, params);

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
