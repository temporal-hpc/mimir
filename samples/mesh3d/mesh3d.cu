#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <cstdint> // uint32_t
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

int main(int argc, char *argv[])
{
    std::string filepath;
    if (argc == 2) { filepath = argv[1]; }
    else
    {
        printf("Usage: ./mesh3d [mesh_file.obj]\n");
        return EXIT_SUCCESS;
    }

    // Parse OBJ file and process errors and warnings, if any
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./";
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filepath, reader_config))
    {
        if (!reader.Error().empty())
        {
            printf("TinyObjReader error: %s\n", reader.Error().c_str());
        }
        return EXIT_FAILURE;
    }
    if (!reader.Warning().empty())
    {
        printf("TinyObjReader warning: %s\n", reader.Warning().c_str());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    printf("Loaded %s: %lu shapes found\n", filepath.c_str(), shapes.size());

    std::vector<float3> h_vertices;
    h_vertices.reserve(attrib.vertices.size()/3);
    std::vector<float3> h_normals;
    h_normals.reserve(attrib.normals.size()/3);
    std::vector<uint32_t> h_triangles;
    // Process a single shape for now
    h_triangles.reserve(shapes[0].mesh.num_face_vertices.size());

    for (size_t vec_start = 0; vec_start < attrib.vertices.size(); vec_start += 3) {
        h_vertices.push_back({
            attrib.vertices[vec_start],
            attrib.vertices[vec_start + 1],
            attrib.vertices[vec_start + 2]
        });
    }

    for (auto shape = shapes.begin(); shape < shapes.end(); ++shape)
    {
        const std::vector<tinyobj::index_t>& indices = shape->mesh.indices;
        const std::vector<int>& material_ids = shape->mesh.material_ids;

        for (size_t index = 0; index < material_ids.size(); ++index)
        {
            // offset by 3 because values are grouped as vertex/normal/texture
            h_triangles.push_back(indices[3 * index + 0].vertex_index);
            h_triangles.push_back(indices[3 * index + 1].vertex_index);
            h_triangles.push_back(indices[3 * index + 2].vertex_index);
        }
    }

    // Check that the collected data is of the expected size
    uint32_t vertex_count = h_vertices.size();
    assert(3 * vertex_count == attrib.vertices.size());
    //assert(h_normals.size() == attrib.normals.size());
    assert(h_triangles.size() == 3 * shapes[0].mesh.num_face_vertices.size());

    ViewerOptions options;
    options.window.size      = {1920,1080}; // Starting window size
    options.background_color = {.5f, .5f, .5f, 1.f};
    EngineHandle engine = nullptr;
    createEngine(options, &engine);

    float *d_coords    = nullptr;
    uint3 *d_triangles = nullptr;

    AllocHandle vertices = nullptr, edges = nullptr;
    auto vert_size = sizeof(float3) * vertex_count;
    auto edge_size = sizeof(uint3) * h_triangles.size();
    allocLinear(engine, (void**)&d_coords, vert_size, &vertices);
    allocLinear(engine, (void**)&d_triangles, edge_size, &edges);

    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc{
        .element_count = vertex_count,
        .view_type     = ViewType::Markers,
        .domain_type   = DomainType::Domain3D,
        .extent        = ViewExtent::make(200,200,200),
    };
    desc.attributes[AttributeType::Position] = {
        .source = vertices,
        .size   = vertex_count,
        .format = FormatDescription::make<float3>(),
    };
    desc.default_size = 1.f;
    createView(engine, &desc, &v1);

    // Recycle the above parameters, changing only what is needed
    desc.element_count = static_cast<uint32_t>(h_triangles.size());
    desc.view_type = ViewType::Edges;
    desc.attributes[AttributeType::Position] = {
        .source     = vertices,
        .size       = vertex_count,
        .format     = FormatDescription::make<float3>(),
        .indices    = edges,
        .index_size = sizeof(uint32_t),
    };
    createView(engine, &desc, &v2);

    checkCuda(cudaMemcpy(d_coords, h_vertices.data(), vert_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_triangles, h_triangles.data(), edge_size, cudaMemcpyHostToDevice));

    displayAsync(engine);

    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_triangles));
    destroyEngine(engine);

    return EXIT_SUCCESS;
}
