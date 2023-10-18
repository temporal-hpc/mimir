#include <cudaview/cudaview.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h" // stbi_load

#include <iostream>
#include <string> // std::string

#include <cudaview/validation.hpp>
using namespace validation; // checkCuda

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

    int width, height, chans;
    auto h_pixels = stbi_load(filepath.c_str(), &width, &height, &chans, STBI_rgb_alpha);
    if (!h_pixels)
    {
        printf("failed to load texture image");
        return EXIT_FAILURE;
    }
    std::cerr << chans << "\n";

    CudaviewEngine engine;
    engine.init(1920, 1080);
    ViewParams params;
    params.element_count = width * height;
    params.extent = {(unsigned)width, (unsigned)height, 1};
    params.data_type = DataType::char4;
    params.data_domain = DataDomain::Domain2D;
    params.resource_type = ResourceType::TextureLinear;
    engine.createView((void**)&d_pixels, params);

    auto tex_size = sizeof(uchar4) * width * height;
    checkCuda(cudaMemcpy(d_pixels, h_pixels, tex_size, cudaMemcpyHostToDevice));
    stbi_image_free(h_pixels);

    engine.updateWindow();
    engine.displayAsync();
    checkCuda(cudaFree(d_pixels));

    return EXIT_SUCCESS;
}
