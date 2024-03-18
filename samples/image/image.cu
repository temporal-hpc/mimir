#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h" // stbi_load

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

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

    MimirEngine engine;
    engine.init(1920, 1080);

    MemoryParams m;
    m.layout         = DataLayout::Layout2D;
    m.element_count  = {(uint)width, (uint)height, 1};
    m.component_type = ComponentType::Char;
    m.channel_count  = 4;
    m.resource_type  = ResourceType::LinearTexture;
    auto pixels = engine.createBuffer((void**)&d_pixels, m);

    ViewParams params;
    params.element_count = width * height;
    params.data_domain   = DataDomain::Domain2D;
    params.domain_type   = DomainType::Structured;
    params.view_type     = ViewType::Image;
    params.attributes[AttributeType::Color] = *pixels;
    /*params.options.external_shaders = {
        {"shaders/texture_vertex2dMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/texture_frag2d_Float4.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };*/
    engine.createView(params);

    auto tex_size = sizeof(uchar4) * width * height;
    checkCuda(cudaMemcpy(d_pixels, h_pixels, tex_size, cudaMemcpyHostToDevice));
    stbi_image_free(h_pixels);

    engine.displayAsync();
    checkCuda(cudaFree(d_pixels));

    return EXIT_SUCCESS;
}
