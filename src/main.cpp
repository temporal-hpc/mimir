#include "cudaview/vk_engine.hpp"

int main()
{
  VulkanEngine engine;
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
  }

  return EXIT_SUCCESS;
}
