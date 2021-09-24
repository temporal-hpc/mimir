#include "cudaview/vk_engine.hpp"
#include "cudaprogram.hpp"

int main()
{
  /*VulkanEngine engine;
  try
  {
    engine.initWindow(800, 600);
    engine.initEngine();
    engine.run();
    engine.cleanup();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }*/

  CudaProgram program;
  program.init(100, 200, 200, 123456);
  for (int i = 0; i < 100; ++i)
  {
    program.runTimestep();
  }
  program.cleanup();

  return EXIT_SUCCESS;
}
