#include "image.hpp"
#include "cuda_utils.hpp"

#include <string> // std::string

#include "stb_image.h" // stbi_load

void ImageProgram::setInitialState()
{
  std::string filename = "textures/lena512.png";
  int channels;
  auto h_pixels = stbi_load(filename.c_str(), &extent.x, &extent.y, &channels,
    STBI_rgb_alpha);
  if (!h_pixels)
  {
    printf("failed to load texture image");
    return;
  }
  auto tex_size = sizeof(uchar4) * extent.x * extent.y;
  checkCuda(cudaMemcpy(d_pixels, h_pixels, tex_size, cudaMemcpyHostToDevice));
  stbi_image_free(h_pixels);
}

void ImageProgram::cleanup()
{
  checkCuda(cudaStreamSynchronize(stream));
  checkCuda(cudaStreamDestroy(stream));
  checkCuda(cudaFree(d_pixels));
}

ImageProgram::ImageProgram()
{
  checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void ImageProgram::runTimestep() {}
