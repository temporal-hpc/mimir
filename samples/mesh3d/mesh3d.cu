#include <cudaview/cudaview.hpp>
#include "cudaview/io.hpp"

#include <curand_kernel.h>

#include <iostream> // std::cerr
#include <string> // std::string
#include <experimental/source_location> // std::experimental::source_location

using source_location = std::experimental::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
    source_location src = source_location::current())
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA assertion: %s on function %s at %s(%d)\n",
        cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
        );
        if (panic) exit(code);
    }
}

int main(int argc, char *argv[])
{
    std::string filepath;
    float *d_coords       = nullptr;
    int3 *d_triangles     = nullptr;
    curandState *d_states = nullptr;
    //unsigned block_size   = 256;

    if (argc == 2)
    {
        filepath = argv[1];
    }
    else
    {
        std::cerr << "Usage: ./mesh3d mesh.off (triangular meshes only)\n";
        return EXIT_FAILURE;
    }

    std::vector<float3> points;
    std::vector<uint3> triangles;
    io::loadTriangleMesh(filepath, points, triangles);
    auto point_count = points.size();

    CudaviewEngine engine;
    engine.init(1920, 1080);
    ViewParams params;
    params.element_count = point_count;
    params.data_type = DataType::float3;
    params.data_domain = DataDomain::Domain3D;
    params.resource_type = ResourceType::UnstructuredBuffer;
    params.primitive_type = PrimitiveType::Points;
    engine.createView((void**)&d_coords, params);

    params.element_count = triangles.size();
    params.data_type = DataType::int3;
    params.primitive_type = PrimitiveType::Edges;
    engine.createView((void**)&d_triangles, params);

    engine.setBackgroundColor({.5f, .5f, .5f, 1.f});

    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
    //unsigned grid_size = (point_count + block_size - 1) / block_size;
    //initSystem<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent, seed);
    //checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(d_coords, points.data(),
        sizeof(float3) * point_count, cudaMemcpyHostToDevice)
    );
    checkCuda(cudaMemcpy(d_triangles, triangles.data(),
        sizeof(uint3) * triangles.size(), cudaMemcpyHostToDevice)
    );

    engine.displayAsync();
    /*for (size_t i = 0; i < iter_count; ++i)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        engine.prepareWindow();
        integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
        engine.updateWindow();
    }*/

    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_triangles));

    return EXIT_SUCCESS;
}
