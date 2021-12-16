#include "meshprogram.hpp"
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

  MeshProgram program;
  try
  {
    // Initialize engine
    VulkanEngine engine;
    engine.init(800, 600);
    engine.registerUnstructuredMemory((void**)&program.d_points,
      9, sizeof(float2), UnstructuredDataType::Points, DataDomain::Domain2D
    );
    engine.registerUnstructuredMemory((void**)&program.d_triangles,
      8, sizeof(uint3), UnstructuredDataType::Edges, DataDomain::Domain2D
    );

    // Cannot make CUDA calls that use the target device memory before
    // registering it on the engine
    program.setInitialState();

    // Set up the function that we want to display
    auto timestep_function = std::bind(&MeshProgram::runTimestep, program);
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
