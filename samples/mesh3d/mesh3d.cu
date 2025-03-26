
#include <cmath> // std::sin
#include <numbers> // std::numbers::pi
#include <iostream> // std::cin

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
#include "mesh_obj.hpp"
using namespace mimir;

float varyAngle(float& deg)
{
    deg += .1f;
    if (deg > 360.f) { deg = 0.f; }
    float rad = deg * (std::numbers::pi_v<float> / 180.f);
    return std::sin(rad);
}

__global__ void breatheKernel(float3 *vertices, float3 *normals, uint32_t vertex_count, float sign)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    for (int i = tidx; i < vertex_count; i += stride)
    {
        auto v = vertices[i];
        auto n = normals[i];
        v.x += 0.001f * sign * n.x;
        v.y += 0.001f * sign * n.y;
        v.z += 0.001f * sign * n.z;
        vertices[i] = v;
    }
}

int main(int argc, char *argv[])
{
    std::string filepath;
    if (argc == 2) { filepath = argv[1]; }
    else
    {
        printf("Usage: ./mesh3d [mesh_file.obj]\n");
        return EXIT_SUCCESS;
    }

    MeshData mesh = loadMeshObj(filepath);
    if (!mesh.loaded)
    {
        printf("Could not load mesh; exiting\n");
        return EXIT_FAILURE;
    }
    uint32_t vertex_count = mesh.vertices.size();
    uint32_t triangle_count = mesh.triangles.size();

    ViewerOptions options;
    options.window.size      = {1920,1080}; // Starting window size
    options.background_color = {.5f, .5f, .5f, 1.f};
    EngineHandle engine = nullptr;
    createEngine(options, &engine);

    float3 *d_coords   = nullptr;
    float3 *d_normals  = nullptr;
    uint3 *d_triangles = nullptr;

    AllocHandle vertices = nullptr, edges = nullptr;
    auto vert_size = sizeof(float3) * vertex_count;
    auto edge_size = sizeof(uint) * triangle_count;
    allocLinear(engine, (void**)&d_coords, vert_size, &vertices);
    allocLinear(engine, (void**)&d_triangles, edge_size, &edges);

    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc{
        .view_type   = ViewType::Markers,
        .domain_type = DomainType::Domain3D,
        .layout      = Layout::make(vertex_count),
    };
    desc.attributes[AttributeType::Position] = {
        .source = vertices,
        .size   = vertex_count,
        .format = FormatDescription::make<float3>(),
    };
    desc.default_size = 1.f;
    desc.linewidth    = 0.f;
    createView(engine, &desc, &v1);

    // Reuse the above parameters, changing only what is needed
    desc.layout    = Layout::make(triangle_count);
    desc.view_type = ViewType::Edges;
    desc.attributes[AttributeType::Position] = {
        .source   = vertices,
        .size     = vertex_count,
        .format   = FormatDescription::make<float3>(),
        .indexing = {
            .source     = edges,
            .size       = triangle_count,
            .index_size = sizeof(uint32_t),
        }
    };
    createView(engine, &desc, &v2);

    checkCuda(cudaMalloc(&d_normals, vert_size));
    checkCuda(cudaMemcpy(d_coords, mesh.vertices.data(), vert_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_triangles, mesh.triangles.data(), edge_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_normals, mesh.normals.data(), vert_size, cudaMemcpyHostToDevice));

    int block_size = 128;
    int grid_size  = (vertex_count + block_size - 1) / block_size;
    float degrees  = 0.f;
    float scale    = varyAngle(degrees);

    displayAsync(engine);
    printf("Press enter to start...\n");
    std::cin.get();
    while (isRunning(engine))
    {
        prepareViews(engine);
        breatheKernel<<< grid_size, block_size >>>(d_coords, d_normals, vertex_count, scale);
        scale = varyAngle(degrees);
        updateViews(engine);
    }

    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_normals));
    checkCuda(cudaFree(d_triangles));
    destroyEngine(engine);

    return EXIT_SUCCESS;
}
