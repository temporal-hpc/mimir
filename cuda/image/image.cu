#include "image.hpp"
#include "cuda_utils.hpp"

#include <string>

#include "stb_image.h"

void ImageProgram::setInitialState()
{
  std::string filename = "textures/lena512.png";
  int channels;
  auto h_image = stbi_load(filename.c_str(), &_extent.x, &_extent.y, &channels,
    STBI_rgb_alpha);
  if (!h_image)
  {
    printf("failed to load texture image");
    return;
  }
  auto tex_size = sizeof(uchar4) * _extent.x * _extent.y;
  checkCuda(cudaMemcpy(_d_image, h_image, tex_size, cudaMemcpyHostToDevice));
  stbi_image_free(h_image);
}

void ImageProgram::cleanup()
{
  checkCuda(cudaStreamSynchronize(_stream));
  checkCuda(cudaStreamDestroy(_stream));
  checkCuda(cudaDeviceReset());
}

ImageProgram::ImageProgram()
{
  checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

void ImageProgram::runTimestep() {}
