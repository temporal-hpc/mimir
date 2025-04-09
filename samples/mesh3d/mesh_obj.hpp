#include <cstdint> // uint32_t
#include <string> // std::string
#include <vector> // std::vector

#include <cuda_runtime_api.h>

struct MeshData
{
    // True if loadMesh finished successfully, false otherwise
    bool loaded;
    // Array of vertex positions
    std::vector<float3> vertices;
    // Array of normal vectors per vertex
    std::vector<float3> normals;
    // Indices array
    std::vector<uint32_t> triangles;
};

MeshData loadMeshObj(const std::string& filepath);