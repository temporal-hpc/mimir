#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h" // stbi_load

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

int main(int argc, char *argv[])
{
    char *filepath = nullptr;
    if (argc == 2)
    {
        filepath = argv[1];
    }
    else
    {
        printf("Usage: ./image path/to/image\n");
        return EXIT_FAILURE;
    }

    uchar4 *d_pixels = nullptr;
    int width, height, chans;
    auto h_pixels = stbi_load(filepath, &width, &height, &chans, STBI_rgb_alpha);
    if (!h_pixels)
    {
        printf("failed to load texture image: %s\n", filepath);
        return EXIT_FAILURE;
    }

    InstanceHandle instance = nullptr;
    createInstance(1920, 1080, &instance);

    AllocHandle pixels;
    allocLinear(instance, (void**)&d_pixels, sizeof(char4) * width * height, &pixels);

    ViewHandle view = nullptr;
    ViewDescription desc{
        .type   = ViewType::Image,
        .domain = DomainType::Domain2D,
        .attributes  = {
            {AttributeType::Position, makeImageFrame(instance)},
            {AttributeType::Color, AttributeDescription{
                .source = pixels,
                .size   = static_cast<unsigned int>(width * height),
                .format = FormatDescription::make<char4>(),
            }}
        },
        .layout       = Layout::make(width, height),
        .default_size = 1.f,
    };
    createView(instance, &desc, &view);

    auto tex_size = sizeof(uchar4) * width * height;
    checkCuda(cudaMemcpy(d_pixels, h_pixels, tex_size, cudaMemcpyHostToDevice));
    stbi_image_free(h_pixels);

    displayAsync(instance);
    checkCuda(cudaFree(d_pixels));
    destroyInstance(instance);

    return EXIT_SUCCESS;
}
