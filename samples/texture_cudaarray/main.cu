#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h" // stbi_load

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

#include <iostream>

// Vector operations copied from 'helper_math.h' at cuda-samples

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);  // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
            ((unsigned int)(rgba.z * 255.0f) << 16) |
            ((unsigned int)(rgba.y * 255.0f) << 8) |
            ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c)
{
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
                                   size_t mip_levels, int filter_radius)
{
    float scale = 1.0f / (float)((filter_radius << 1) + 1);
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < baseHeight)
    {
        for (uint32_t mipLevelIdx = 0; mipLevelIdx < mip_levels; mipLevelIdx++)
        {
            uint32_t width = (baseWidth >> mipLevelIdx) ? (baseWidth >> mipLevelIdx) : 1;
            uint32_t height = (baseHeight >> mipLevelIdx) ? (baseHeight >> mipLevelIdx) : 1;
            if (y < height && filter_radius < width)
            {
                float px = 1.0 / width;
                float py = 1.0 / height;
                float4 t = make_float4(0.0f);
                for (int x = -filter_radius; x <= filter_radius; x++)
                {
                    t += tex2DLod<float4>(textureMipMapInput, x * px, y * py, (float)mipLevelIdx);
                }

                unsigned int dataB = rgbaFloatToInt(t * scale);
                surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], 0, y);

                for (int x = 1; x < width; x++)
                {
                    t += tex2DLod<float4>(textureMipMapInput,
                        (x + filter_radius) * px, y * py, (float)mipLevelIdx);
                    t -= tex2DLod<float4>(textureMipMapInput,
                        (x - filter_radius - 1) * px, y * py, (float)mipLevelIdx);
                    unsigned int dataB = rgbaFloatToInt(t * scale);
                    surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], x * sizeof(uchar4), y);
                }
            }
        }
    }
}

// column pass using coalesced global memory reads
__global__ void d_boxfilter_rgba_y(cudaSurfaceObject_t* dstSurfMipMapArray,
                                   cudaSurfaceObject_t* srcSurfMipMapArray,
                                   size_t baseWidth, size_t baseHeight,
                                   size_t mip_levels, int filter_radius)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1.0f / (float)((filter_radius << 1) + 1);

    for (uint32_t mipLevelIdx = 0; mipLevelIdx < mip_levels; mipLevelIdx++)
    {
        uint32_t width = (baseWidth >> mipLevelIdx) ? (baseWidth >> mipLevelIdx) : 1;
        uint32_t height = (baseHeight >> mipLevelIdx) ? (baseHeight >> mipLevelIdx) : 1;

        if (x < width && height > filter_radius)
        {
            float4 t;
            // do left edge
            int colInBytes = x * sizeof(uchar4);
            unsigned int pixFirst = surf2Dread<unsigned int>(
                srcSurfMipMapArray[mipLevelIdx], colInBytes, 0);
            t = rgbaIntToFloat(pixFirst) * filter_radius;

            for (int y = 0; (y < (filter_radius + 1)) && (y < height); y++)
            {
                unsigned int pix = surf2Dread<unsigned int>(
                    srcSurfMipMapArray[mipLevelIdx], colInBytes, y);
                t += rgbaIntToFloat(pix);
            }

            unsigned int dataB = rgbaFloatToInt(t * scale);
            surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], colInBytes, 0);

            for (int y = 1; (y < filter_radius + 1) && ((y + filter_radius) < height); y++)
            {
                unsigned int pix = surf2Dread<unsigned int>(
                    srcSurfMipMapArray[mipLevelIdx], colInBytes, y + filter_radius);
                t += rgbaIntToFloat(pix);
                t -= rgbaIntToFloat(pixFirst);

                dataB = rgbaFloatToInt(t * scale);
                surf2Dwrite(dataB, dstSurfMipMapArray[mipLevelIdx], colInBytes, y);
            }

            // main loop
            for (int y = (filter_radius + 1); y < (height - filter_radius); y++)
            {
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
            for (int y = height - filter_radius; (y < height) && ((y - filter_radius - 1) > 1); y++)
            {
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
int mip_levels    = 1;

// This varies the filter radius, so we can see automatic animation
void varySigma()
{
    filter_radius += g_nFilterSign;

    if (filter_radius > 64)
    {
        filter_radius = 64;  // clamp to 64 and then negate sign
        g_nFilterSign = -1;
    }
    else if (filter_radius < 0)
    {
        filter_radius = 0;
        g_nFilterSign = 1;
    }
}

int main(int argc, char *argv[])
{
    std::string filepath;
    if (argc == 2)
    {
        filepath = argv[1];
    }
    else
    {
        printf("Usage: ./run_texture path/to/image\n");
        return EXIT_FAILURE;
    }

    int img_width, img_height, chans;
    auto img_data = stbi_load(filepath.c_str(), &img_width, &img_height, &chans, STBI_rgb_alpha);
    if (!img_data)
    {
        printf("failed to load texture image");
        return EXIT_FAILURE;
    }
    printf("Loaded '%s', '%d'x'%d pixels \n", filepath.c_str(), img_width, img_height);

    ViewerOptions options;
    options.window.size = {1920,1080}; // Starting window size
    options.present     = { .mode = PresentMode::VSync };
    InstanceHandle instance = nullptr;
    createInstance(options, &instance);

    cudaMipmappedArray_t mipmap_array = nullptr;
    cudaChannelFormatDesc cuda_format{
        .x = 8, .y = 8, .z = 8, .w = 8,
        .f = cudaChannelFormatKindUnsigned,
    };
    auto cuda_extent = make_cudaExtent(img_width, img_height, 0);
    AllocHandle mipmap;
    allocMipmap(instance, &mipmap_array, &cuda_format, cuda_extent, mip_levels, &mipmap);
    TextureDescription tex{
        .source = mipmap,
        .format = FormatDescription::make<uchar4>(),
        .extent = Layout::make(img_width, img_height, 1),
        .levels = static_cast<unsigned int>(mip_levels),
    };
    copyTextureData(instance, tex, img_data, sizeof(char4) * img_width * img_height);

    ViewHandle view = nullptr;
    ViewDescription desc{
        .type   = ViewType::Image,
        .domain = DomainType::Domain2D,
        .attributes  = {
            { AttributeType::Position, makeImageFrame(instance) },
            { AttributeType::Color, {
                .source = tex.source,
                .size   = static_cast<unsigned int>(img_width * img_height),
                .format = tex.format,
            }}
        },
        .layout = Layout::make(img_width, img_height),
    };
    createView(instance, &desc, &view);

    cudaMipmappedArray_t cudaMipmappedImageArrayTemp = nullptr;
    checkCuda(cudaMallocMipmappedArray(
        &cudaMipmappedImageArrayTemp, &cuda_format, cuda_extent, mip_levels
    ));
    cudaMipmappedArray_t cudaMipmappedImageArrayOrig = nullptr;
    checkCuda(cudaMallocMipmappedArray(
        &cudaMipmappedImageArrayOrig, &cuda_format, cuda_extent, mip_levels
    ));

    cudaResourceDesc res_desc{
        .resType = cudaResourceTypeMipmappedArray,
        .res     = { .mipmap = { .mipmap = cudaMipmappedImageArrayOrig }},
    };

    cudaTextureDesc tex_desc{
        .addressMode         = { cudaAddressModeWrap },
        .filterMode          = cudaFilterModeLinear,
        .readMode            = cudaReadModeNormalizedFloat,
        .normalizedCoords    = true,
        .mipmapFilterMode    = cudaFilterModeLinear,
        .maxMipmapLevelClamp = static_cast<float>(mip_levels - 1),
    };

    cudaTextureObject_t tex_obj = 0;
    checkCuda(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    std::vector<cudaSurfaceObject_t> surf_obj_list, surf_obj_list_temp;
    for (int level_idx = 0; level_idx < mip_levels; ++level_idx)
    {
        cudaArray_t mipLevelArray, mipLevelArrayTemp, mipLevelArrayOrig;

        checkCuda(cudaGetMipmappedArrayLevel(
            &mipLevelArray, mipmap_array, level_idx
        ));
        checkCuda(cudaGetMipmappedArrayLevel(
            &mipLevelArrayTemp, cudaMipmappedImageArrayTemp, level_idx
        ));
        checkCuda(cudaGetMipmappedArrayLevel(
            &mipLevelArrayOrig, cudaMipmappedImageArrayOrig, level_idx
        ));

        uint32_t width = (img_width >> level_idx) ? (img_width >> level_idx) : 1;
        uint32_t height = (img_height >> level_idx) ? (img_height >> level_idx) : 1;
        checkCuda(cudaMemcpy2DArrayToArray(
            mipLevelArrayOrig, 0, 0, mipLevelArray, 0, 0,
            width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice
        ));

        cudaResourceDesc res_desc{
            .resType = cudaResourceTypeArray,
            .res     = { .array { .array = mipLevelArray } },
        };
        cudaSurfaceObject_t surf_obj = 0;
        checkCuda(cudaCreateSurfaceObject(&surf_obj, &res_desc));
        surf_obj_list.push_back(surf_obj);

        cudaResourceDesc res_desc_temp{
            .resType = cudaResourceTypeArray,
            .res     = { .array { .array = mipLevelArrayTemp } },
        };
        cudaSurfaceObject_t surf_temp = 0;
        checkCuda(cudaCreateSurfaceObject(&surf_temp, &res_desc_temp));
        surf_obj_list_temp.push_back(surf_temp);
    }

    cudaSurfaceObject_t *d_surf_list = nullptr;
    checkCuda(cudaMalloc(&d_surf_list, sizeof(cudaSurfaceObject_t) * mip_levels));
    checkCuda(cudaMemcpy(d_surf_list, surf_obj_list.data(),
        sizeof(cudaSurfaceObject_t) * mip_levels, cudaMemcpyHostToDevice
    ));
    cudaSurfaceObject_t *d_surf_list_temp = nullptr;
    checkCuda(cudaMalloc(&d_surf_list_temp, sizeof(cudaSurfaceObject_t) * mip_levels));
    checkCuda(cudaMemcpy(d_surf_list_temp, surf_obj_list_temp.data(),
        sizeof(cudaSurfaceObject_t) * mip_levels, cudaMemcpyHostToDevice
    ));

    int nthreads = 128;

    // instance does not run before starting display
    assert(!isRunning(instance));
    displayAsync(instance);
    while (isRunning(instance))
    {
        prepareViews(instance);
        // Perform 2D box filter on image using CUDA
        d_boxfilter_rgba_x<<<img_height / nthreads, nthreads >>>(
            d_surf_list_temp, tex_obj, img_width, img_height, mip_levels, filter_radius
        );
        d_boxfilter_rgba_y<<<img_width / nthreads, nthreads >>>(
            d_surf_list, d_surf_list_temp, img_width, img_height, mip_levels, filter_radius
        );
        varySigma();
        updateViews(instance);
    }

    checkCuda(cudaDestroyTextureObject(tex_obj));
    checkCuda(cudaFreeMipmappedArray(cudaMipmappedImageArrayTemp));
    checkCuda(cudaFreeMipmappedArray(cudaMipmappedImageArrayOrig));
    destroyInstance(instance);

    return EXIT_SUCCESS;
}
