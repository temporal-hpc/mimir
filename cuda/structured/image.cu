#include "image.hpp"
#include "cuda_utils.hpp"

#include <chrono>
#include <random>

#include "stb_image.h"

void ImageProgram::setInitialState()
{
  checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
  std::string filename = "textures/lena512.png";
  int channels;
  auto h_image = stbi_load(filename.c_str(), &_extent.x, &_extent.y, &channels,
    STBI_rgb_alpha);
  if (!h_image)
  {
    printf("failed to load texture image");
    return;
  }
  size_t tex_size = _extent.x * _extent.y * 4 * sizeof(unsigned char);
  checkCuda(cudaMemcpy(_d_image, h_image, tex_size, cudaMemcpyHostToDevice));
  stbi_image_free(h_image);
}

void ImageProgram::cleanup()
{
  checkCuda(cudaStreamSynchronize(_stream));
  checkCuda(cudaStreamDestroy(_stream));
  checkCuda(cudaDeviceReset());
}

ImageProgram::ImageProgram() {}
void ImageProgram::runTimestep() {}
