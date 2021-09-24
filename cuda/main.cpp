#include "cudaprogram.hpp"
#include "cudaview/vk_engine.hpp"

int main()
{

  CudaProgram program;
  program.init(100, 200, 200, 123456);
  auto func = std::bind(&CudaProgram::runTimestep, program);

  VulkanEngine engine;
  try
  {
    engine.init();
    engine.run(func, 100);
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    engine.cleanup();
    program.cleanup();
    return EXIT_FAILURE;
  }
  engine.cleanup();
  program.cleanup();

  return EXIT_SUCCESS;
}
