#include <fstream> // std::ifstream
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

// Reads an .off file and only loads the points and triangles
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
        printf("Usage: ./mesh3d mesh.off (triangular meshes only)\n");
        return EXIT_FAILURE;
    }

    std::vector<float3> h_points;
    std::vector<uint3> h_triangles;
    loadTriangleMesh(filepath, h_points, h_triangles);
    auto point_count = h_points.size();

    MimirEngine engine;
    engine.init(1920, 1080);
    engine.setBackgroundColor({.5f, .5f, .5f, 1.f});

    auto vertices = engine.allocateMemory((void**)&d_coords, sizeof(float3) * point_count);
    auto edges    = engine.allocateMemory((void**)&d_triangles, sizeof(int3) * h_triangles.size());

    ViewParams2 params;
    params.element_count = point_count;
    params.data_domain   = DataDomain::Domain3D;
    params.view_type     = ViewType::Markers;
    params.options.default_size = 20.f;
    params.attributes[AttributeType::Position] = {
        .memory = vertices,
        .format = { .type = DataType::float32, .components = 3 }
    };
    engine.createView(params);

    // Recycle the above parameters, changing only what is needed
    params.element_count = 3 * h_triangles.size();
    params.view_type     = ViewType::Edges;
    params.indexing = {
        .memory = edges,
        .format = { .type = DataType::int32, .components = 1 }
    };
    engine.createView(params);

    checkCuda(cudaMemcpy(d_coords, h_points.data(),
        sizeof(float3) * point_count, cudaMemcpyHostToDevice)
    );
    checkCuda(cudaMemcpy(d_triangles, h_triangles.data(),
        sizeof(uint3) * h_triangles.size(), cudaMemcpyHostToDevice)
    );

    engine.displayAsync();

    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_triangles));

    return EXIT_SUCCESS;
}
