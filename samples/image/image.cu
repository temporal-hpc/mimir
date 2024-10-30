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

    Engine engine = nullptr;
    createEngine(1920, 1080, &engine);

    AllocHandle pixels = nullptr;
    allocLinear(engine, (void**)&d_pixels, sizeof(char4) * width * height, &pixels);

    TextureHandle teximg = nullptr;
    TextureDescription tex_desc
    {
        .source = pixels,
        .format = FormatDescription::make<char4>(),
        .extent = ViewExtent::make(width, height, 1),
        .levels = 1,
    };
    makeTexture(engine, tex_desc, &teximg);

    ViewHandle view = nullptr;
    ViewDescription desc{
        .element_count = 6, //static_cast<unsigned int>(width * height),
        .view_type     = ViewType::Markers,
        .domain_type   = DomainType::Domain2D,
        .extent        = ViewExtent::make(1, 1, 1),
        .attributes    = {},
        .textures      = {},
    };
    desc.attributes[AttributeType::Position] = makeImageFrame(engine);
    createView(engine, &desc, &view);
    view->default_size = 1.f;

    /*MemoryParams m;
    m.layout         = DataLayout::Layout2D;
    m.element_count  = {(uint)width, (uint)height, 1};
    m.component_type = ComponentType::Char;
    m.channel_count  = 4;
    m.resource_type  = ResourceType::LinearTexture;
    auto pixels = engine->createBuffer((void**)&d_pixels, m);

    ViewParamsOld params;
    params.element_count = width * height;
    params.data_domain   = DomainType::Domain2D;
    params.domain_type   = DomainType::Structured;
    params.view_type     = ViewType::Image;
    params.attributes[AttributeType::Color] = *pixels;
    /*params.options.external_shaders = {
        {"shaders/texture_vertex2dMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/texture_frag2d_Float4.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };
    engine->createView(params);*/

    auto tex_size = sizeof(uchar4) * width * height;
    checkCuda(cudaMemcpy(d_pixels, h_pixels, tex_size, cudaMemcpyHostToDevice));
    stbi_image_free(h_pixels);

    displayAsync(engine);
    checkCuda(cudaFree(d_pixels));
    destroyEngine(engine);

    return EXIT_SUCCESS;
}
