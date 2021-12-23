#include "cudaview/vk_engine.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h" // stbi_load

#include <iostream>
#include <string> // std::string

#include "cuda_utils.hpp"

int main(int argc, char *argv[])
{
  std::string filepath;
  if (argc == 2)
  {
    filepath = argv[1];
  }
  else
  {
    std::cerr << "Usage: ./image path/to/image\n";
    return EXIT_FAILURE;
  }

  uchar4 *d_pixels = nullptr;

  int channels, width, height;
  auto h_pixels = stbi_load(filepath.c_str(), &width, &height, &channels,
    STBI_rgb_alpha);
  if (!h_pixels)
  {
    printf("failed to load texture image");
    return EXIT_FAILURE;
  }

  VulkanEngine engine;
  engine.init(900, 900);
  engine.registerStructuredMemory((void**)&d_pixels,
    width, height, sizeof(uchar4), DataFormat::Rgba32
  );

  auto tex_size = sizeof(uchar4) * width * height;
  checkCuda(cudaMemcpy(d_pixels, h_pixels, tex_size, cudaMemcpyHostToDevice));
  stbi_image_free(h_pixels);

  engine.displayAsync();
  checkCuda(cudaFree(d_pixels));

  return EXIT_SUCCESS;
}
