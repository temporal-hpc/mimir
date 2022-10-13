#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <chrono> // std::chrono
#include <thread> // std::thread

#include "helper_image.h"
#include "helper_math.h"
#include "cudaview/vk_engine.hpp"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return ((unsigned int)(rgba.w * 255.0f) << 24) |
         ((unsigned int)(rgba.z * 255.0f) << 16) |
         ((unsigned int)(rgba.y * 255.0f) << 8) |
         ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c) {
  float4 rgba;
  rgba.x = (c & 0xff) * 0.003921568627f;          //  /255.0f;
  rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;   //  /255.0f;
  rgba.z = ((c >> 16) & 0xff) * 0.003921568627f;  //  /255.0f;
  rgba.w = ((c >> 24) & 0xff) * 0.003921568627f;  //  /255.0f;
  return rgba;
}

// row pass using texture lookups
__global__ void d_boxfilter_rgba_x(cudaSurfaceObject_t* dstSurfMipMapArray,
                                   cudaTextureObject_t textureMipMapInput,
                                   size_t baseWidth, size_t baseHeight,
                                   size_t mipLevels, int filter_radius) {
  float scale = 1.0f / (float)((filter_radius << 1) + 1);
  unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

  if (y < baseHeight) {
    for (uint32_t mipLevelIdx = 0; mipLevelIdx < mipLevels; mipLevelIdx++) {
      uint32_t width =
          (baseWidth >> mipLevelIdx) ? (baseWidth >> mipLevelIdx) : 1;
      uint32_t height =
          (baseHeight >> mipLevelIdx) ? (baseHeight >> mipLevelIdx) : 1;
      if (y < height && filter_radius < width) {
        float px = 1.0 / width;
        float py = 1.0 / height;
        float4 t = make_float4(0.0f);
        for (int x = -filter_radius; x <= filter_radius; x++) {
          t += tex2DLod<float4>(textureMipMapInput, x * px, y * py,
                                (float)mipLevelIdx);
        }

        unsigned int dataB = rgbaFloatToInt(t * scale);
        surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], 0, y);

        for (int x = 1; x < width; x++) {
          t += tex2DLod<float4>(textureMipMapInput, (x + filter_radius) * px,
                                y * py, (float)mipLevelIdx);
          t -=
              tex2DLod<float4>(textureMipMapInput, (x - filter_radius - 1) * px,
                               y * py, (float)mipLevelIdx);
          unsigned int dataB = rgbaFloatToInt(t * scale);
          surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx],
                      x * sizeof(uchar4), y);
        }
      }
    }
  }
}

// column pass using coalesced global memory reads
__global__ void d_boxfilter_rgba_y(cudaSurfaceObject_t* dstSurfMipMapArray,
                                   cudaSurfaceObject_t* srcSurfMipMapArray,
                                   size_t baseWidth, size_t baseHeight,
                                   size_t mipLevels, int filter_radius) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  float scale = 1.0f / (float)((filter_radius << 1) + 1);

  for (uint32_t mipLevelIdx = 0; mipLevelIdx < mipLevels; mipLevelIdx++) {
    uint32_t width =
        (baseWidth >> mipLevelIdx) ? (baseWidth >> mipLevelIdx) : 1;
    uint32_t height =
        (baseHeight >> mipLevelIdx) ? (baseHeight >> mipLevelIdx) : 1;

    if (x < width && height > filter_radius) {
      float4 t;
      // do left edge
      int colInBytes = x * sizeof(uchar4);
      unsigned int pixFirst = surf2Dread<unsigned int>(
          srcSurfMipMapArray[mipLevelIdx], colInBytes, 0);
      t = rgbaIntToFloat(pixFirst) * filter_radius;

      for (int y = 0; (y < (filter_radius + 1)) && (y < height); y++) {
        unsigned int pix = surf2Dread<unsigned int>(
            srcSurfMipMapArray[mipLevelIdx], colInBytes, y);
        t += rgbaIntToFloat(pix);
      }

      unsigned int dataB = rgbaFloatToInt(t * scale);
      surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], colInBytes, 0);

      for (int y = 1; (y < filter_radius + 1) && ((y + filter_radius) < height);
           y++) {
        unsigned int pix = surf2Dread<unsigned int>(
            srcSurfMipMapArray[mipLevelIdx], colInBytes, y + filter_radius);
        t += rgbaIntToFloat(pix);
        t -= rgbaIntToFloat(pixFirst);

        dataB = rgbaFloatToInt(t * scale);
        surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], colInBytes, y);
      }

      // main loop
      for (int y = (filter_radius + 1); y < (height - filter_radius); y++) {
        unsigned int pix = surf2Dread<unsigned int>(
            srcSurfMipMapArray[mipLevelIdx], colInBytes, y + filter_radius);
        t += rgbaIntToFloat(pix);

        pix = surf2Dread<unsigned int>(srcSurfMipMapArray[mipLevelIdx],
                                       colInBytes, y - filter_radius - 1);
        t -= rgbaIntToFloat(pix);

        dataB = rgbaFloatToInt(t * scale);
        surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], colInBytes, y);
      }

      // do right edge
      unsigned int pixLast = surf2Dread<unsigned int>(
          srcSurfMipMapArray[mipLevelIdx], colInBytes, height - 1);
      for (int y = height - filter_radius;
           (y < height) && ((y - filter_radius - 1) > 1); y++) {
        t += rgbaIntToFloat(pixLast);
        unsigned int pix = surf2Dread<unsigned int>(
            srcSurfMipMapArray[mipLevelIdx], colInBytes, y - filter_radius - 1);
        t -= rgbaIntToFloat(pix);
        dataB = rgbaFloatToInt(t * scale);
        surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], colInBytes, y);
      }
    }
  }
}

int filter_radius = 14;
int g_nFilterSign = 1;
int mipLevels     = 1;

// This varies the filter radius, so we can see automatic animation
void varySigma() {
  filter_radius += g_nFilterSign;

  if (filter_radius > 64) {
    filter_radius = 64;  // clamp to 64 and then negate sign
    g_nFilterSign = -1;
  } else if (filter_radius < 0) {
    filter_radius = 0;
    g_nFilterSign = 1;
  }
}

int main(int argc, char *argv[])
{
  int width = 900, height = 900;
  VulkanEngine engine;
  engine.init(width, height);

  // TODO: Load image
  std::string filename = "teapot1024.ppm";
  unsigned *img_data  = nullptr;
  unsigned img_width  = 0;
  unsigned img_height = 0;
  sdkLoadPPM4(filename.c_str(), (unsigned char**)&img_data, &img_width, &img_height);
  printf("Loaded '%s', '%d'x'%d pixels \n", filename.c_str(), img_width, img_height);

  uchar4 *d_dummy = nullptr;

  ViewParams params;
  params.element_count  = img_width * img_height;
  params.element_size   = sizeof(uchar4);
  params.data_domain    = DataDomain::Domain2D;
  params.resource_type  = ResourceType::Texture;
  params.texture_format = TextureFormat::Rgba32;
  auto view = engine.addView((void**)&d_dummy, params);

  engine.displayAsync();

  int nthreads = 128;
  for (int i = 0; i < 999999; i++)
  {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    engine.prepareWindow();

    // Perform 2D box filter on image using CUDA
    d_boxfilter_rgba_x<<<img_height / nthreads, nthreads >>>(
      view.d_surfaceObjectListTemp, view.textureObjMipMapInput,
      img_width, img_height, mipLevels, filter_radius
    );
    d_boxfilter_rgba_y<<<img_width / nthreads, nthreads >>>(
      view.d_surfaceObjectList, view.d_surfaceObjectListTemp,
      img_width, img_height, mipLevels, filter_radius
    );
    varySigma();

    engine.updateWindow();
  }

  /*CUDA_CALL(cudaFree(dStates));
  CUDA_CALL(cudaFree(dPoints));
  free(hPoints);*/

  return EXIT_SUCCESS;
}
