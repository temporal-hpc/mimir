#include "cudaprogram.hpp"
#include "cudaview/vk_engine.hpp"

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
    VulkanEngine engine(program.stream);
    engine.init(800, 600);
    ViewParams params;
    params.element_count = program.particle_count;
    params.element_size = sizeof(float2);
    params.extent = {200, 200, 1};
    params.data_domain = DataDomain::Domain2D;
    params.resource_type = ResourceType::Buffer;
    params.primitive_type = PrimitiveType::Points;
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
