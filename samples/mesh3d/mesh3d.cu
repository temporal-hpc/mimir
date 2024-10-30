#include <fstream> // std::ifstream
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

// Load an .off mesh file, only reading vertex and triangle data
void loadTriangleMesh(const std::string& filename,
    std::vector<float3>& points, std::vector<uint3>& triangles)
{
    constexpr auto stream_max = std::numeric_limits<std::streamsize>::max();
    std::string temp_string;

    std::ifstream stream(filename);
    stream >> temp_string;
    if (temp_string.compare("OFF") != 0)
    {
        return;
    }
    stream.ignore(stream_max, '\n');
    while (stream.peek() == '#')
    {
        stream.ignore(stream_max, '\n');
    }

    size_t point_count, face_count, edge_count;
    stream >> point_count >> face_count >> edge_count;
    points.reserve(point_count);
    triangles.reserve(edge_count);

    for (size_t i = 0; i < point_count; ++i)
    {
        float3 point;
        stream >> point.x >> point.y >> point.z;
        points.push_back(point);
    }
    for (size_t j = 0; j < face_count; ++j)
    {
        uint3 triangle;
        stream >> temp_string >> triangle.x >> triangle.y >> triangle.z;
        triangles.push_back(triangle);
        stream.ignore(stream_max, '\n');
    }
}

int main(int argc, char *argv[])
{
    std::string filepath;
    float *d_coords       = nullptr;
    int3 *d_triangles     = nullptr;

    if (argc == 2)
    {
        filepath = argv[1];
    }
    else
    {
        printf("Usage: ./mesh3d mesh.off (triangle meshes only)\n");
        return EXIT_FAILURE;
    }

    std::vector<float3> h_points;
    std::vector<uint3> h_triangles;
    loadTriangleMesh(filepath, h_points, h_triangles);
    unsigned int point_count = h_points.size();

    ViewerOptions options;
    options.window.size = {1920,1080}; // Starting window size
    options.bg_color    = {.5f, .5f, .5f, 1.f};
    Engine engine = nullptr;
    createEngine(options, &engine);

    AllocHandle vertices = nullptr, edges = nullptr;
    auto vert_size = sizeof(float3) * point_count;
    auto edge_size = sizeof(int3) * h_triangles.size();
    allocLinear(engine, (void**)&d_coords, vert_size, &vertices);
    allocLinear(engine, (void**)&d_triangles, edge_size, &edges);

    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc{
        .element_count = point_count,
        .view_type     = ViewType::Markers,
        .domain_type   = DomainType::Domain3D,
        .extent        = ViewExtent::make(1,1,1),
    };
    desc.attributes[AttributeType::Position] = {
        .source = vertices,
        .size   = point_count,
        .format = FormatDescription::make<float3>(),
    };
    printf("Creating v1, elements %u, array size %lu\n", desc.element_count, vert_size);
    createView(engine, &desc, &v1);
    v1->default_size = 20.f;

    // Recycle the above parameters, changing only what is needed
    desc.element_count = static_cast<unsigned int>(3 * h_triangles.size());
    desc.view_type = ViewType::Edges;
    desc.attributes[AttributeType::Position] = {
        .source     = vertices, // TODO: Consider this instead of element_count when creating view resources
        .size       = point_count,
        .format     = FormatDescription::make<float3>(),
        .indices    = edges,
        .index_size = sizeof(int),
    };
    printf("Creating v2, elements %u, array size %lu\n", desc.element_count, edge_size);
    createView(engine, &desc, &v2);

    checkCuda(cudaMemcpy(d_coords, h_points.data(), vert_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_triangles, h_triangles.data(), edge_size, cudaMemcpyHostToDevice));

    displayAsync(engine);

    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_triangles));
    destroyEngine(engine);

    return EXIT_SUCCESS;
}
