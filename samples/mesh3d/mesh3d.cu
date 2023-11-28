#include <fstream> // std::ifstream
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

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

    std::vector<float3> points;
    std::vector<uint3> triangles;
    loadTriangleMesh(filepath, points, triangles);
    auto point_count = points.size();

    CudaviewEngine engine;
    engine.init(1920, 1080);
    engine.setBackgroundColor({.5f, .5f, .5f, 1.f});

    MemoryParams m;
    m.layout          = DataLayout::Layout1D;
    m.element_count.x = point_count;
    m.data_type       = DataType::Float;
    m.channel_count   = 3;
    m.resource_type   = ResourceType::Buffer;
    auto pointsmem = engine.createBuffer((void**)&d_coords, m);

    m.element_count.x = 3 * triangles.size();
    m.data_type       = DataType::Int;
    m.channel_count   = 1;
    m.resource_type   = ResourceType::IndexBuffer;
    auto trimem = engine.createBuffer((void**)&d_triangles, m);

    ViewParams params;
    params.element_count = point_count;
    params.data_domain   = DataDomain::Domain3D;
    params.domain_type   = DomainType::Unstructured;
    params.view_type     = ViewType::Markers;
    params.attributes[AttributeType::Position] = *pointsmem;
    engine.createView(params);

    params.element_count = triangles.size();
    params.view_type     = ViewType::Edges;
    params.attributes[AttributeType::Index] = *trimem;
    engine.createView(params);

    checkCuda(cudaMemcpy(d_coords, points.data(),
        sizeof(float3) * point_count, cudaMemcpyHostToDevice)
    );
    checkCuda(cudaMemcpy(d_triangles, triangles.data(),
        sizeof(uint3) * triangles.size(), cudaMemcpyHostToDevice)
    );

    engine.displayAsync();

    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_triangles));

    return EXIT_SUCCESS;
}
