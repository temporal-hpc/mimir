#include "cudaprogram.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <functional> // TODO: Use bound function in engine.mainLoop()

int main()
{
  CudaProgram program;
  program.init(100, 200, 200, 123456);
  auto func = std::bind(&CudaProgram::runTimestep, program);

  VulkanEngine engine;
  try
  {
    engine.init();
    engine.mainLoop();
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
