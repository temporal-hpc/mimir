#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <cstdint> // uint32_t
#include <string> // std::string
#include <vector> // std::vector

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

__global__ void breatheKernel(float3 *vertices, uint32_t vertex_count)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    for (int i = tidx; i < vertex_count; i += stride)
    {
        auto v = vertices[i];
    }
}

// This varies the filter radius, so we can see automatic animation
void varySigma(int& radius, int& sign)
{
    radius += sign;

    if (radius > 64)
    {
        radius = 64;  // clamp to 64 and then negate sign
        sign = -1;
    }
    else if (radius < 0)
    {
        radius = 0;
        sign = 1;
    }
}

struct vec3
{
    float v[3];
    vec3() {
        v[0] = 0.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
    }
};

static void computeAllSmoothingNormals(tinyobj::attrib_t& attrib,
    std::vector<tinyobj::shape_t>& shapes)
{
    vec3 p[3];
    for (size_t s = 0, slen = shapes.size(); s < slen; ++s)
    {
        const tinyobj::shape_t& shape(shapes[s]);
        size_t facecount = shape.mesh.num_face_vertices.size();
        assert(shape.mesh.smoothing_group_ids.size());

        for (size_t f = 0, flen = facecount; f < flen; ++f)
        {
            for (unsigned int v = 0; v < 3; ++v)
            {
                tinyobj::index_t idx = shape.mesh.indices[3*f + v];
                assert(idx.vertex_index != -1);
                p[v].v[0] = attrib.vertices[3*idx.vertex_index  ];
                p[v].v[1] = attrib.vertices[3*idx.vertex_index+1];
                p[v].v[2] = attrib.vertices[3*idx.vertex_index+2];
            }

            // cross(p[1] - p[0], p[2] - p[0])
            float nx = (p[1].v[1] - p[0].v[1]) * (p[2].v[2] - p[0].v[2]) -
            (p[1].v[2] - p[0].v[2]) * (p[2].v[1] - p[0].v[1]);
            float ny = (p[1].v[2] - p[0].v[2]) * (p[2].v[0] - p[0].v[0]) -
            (p[1].v[0] - p[0].v[0]) * (p[2].v[2] - p[0].v[2]);
            float nz = (p[1].v[0] - p[0].v[0]) * (p[2].v[1] - p[0].v[1]) -
            (p[1].v[1] - p[0].v[1]) * (p[2].v[0] - p[0].v[0]);

            // Don't normalize here.
            for (unsigned int v = 0; v < 3; ++v)
            {
                tinyobj::index_t idx = shape.mesh.indices[3*f + v];
                attrib.normals[3*idx.normal_index  ] += nx;
                attrib.normals[3*idx.normal_index+1] += ny;
                attrib.normals[3*idx.normal_index+2] += nz;
            }
        }
    }

    assert(attrib.normals.size() % 3 == 0);
    for (size_t i = 0, nlen = attrib.normals.size() / 3; i < nlen; ++i)
    {
        tinyobj::real_t& nx = attrib.normals[3*i  ];
        tinyobj::real_t& ny = attrib.normals[3*i+1];
        tinyobj::real_t& nz = attrib.normals[3*i+2];
        tinyobj::real_t len = sqrtf(nx*nx + ny*ny + nz*nz);
        tinyobj::real_t scale = len == 0 ? 0 : 1 / len;
        nx *= scale;
        ny *= scale;
        nz *= scale;
    }
}

static void computeSmoothingShape(tinyobj::attrib_t& inattrib, tinyobj::shape_t& inshape,
    std::vector<std::pair<unsigned int, unsigned int>>& sortedids,
    unsigned int idbegin, unsigned int idend,
    std::vector<tinyobj::shape_t>& outshapes,
    tinyobj::attrib_t& outattrib)
{
    unsigned int sgroupid = sortedids[idbegin].first;
    bool hasmaterials = inshape.mesh.material_ids.size();
    // Make a new shape from the set of faces in the range [idbegin, idend).
    outshapes.emplace_back();
    tinyobj::shape_t& outshape = outshapes.back();
    outshape.name = inshape.name;
    // Skip lines and points.

    std::unordered_map<unsigned int, unsigned int> remap;
    for (unsigned int id = idbegin; id < idend; ++id)
    {
        unsigned int face = sortedids[id].second;

        outshape.mesh.num_face_vertices.push_back(3); // always triangles
        if (hasmaterials) { outshape.mesh.material_ids.push_back(inshape.mesh.material_ids[face]); }
        outshape.mesh.smoothing_group_ids.push_back(sgroupid);
        // Skip tags.

        for (unsigned int v = 0; v < 3; ++v)
        {
            tinyobj::index_t inidx = inshape.mesh.indices[3*face + v], outidx;
            assert(inidx.vertex_index != -1);
            auto iter = remap.find(inidx.vertex_index);
            // Smooth group 0 disables smoothing so no shared vertices in that case.
            if (sgroupid && iter != remap.end())
            {
                outidx.vertex_index = (*iter).second;
                outidx.normal_index = outidx.vertex_index;
                outidx.texcoord_index = (inidx.texcoord_index == -1) ? -1 : outidx.vertex_index;
            }
            else
            {
                assert(outattrib.vertices.size() % 3 == 0);
                unsigned int offset = static_cast<unsigned int>(outattrib.vertices.size() / 3);
                outidx.vertex_index = outidx.normal_index = offset;
                outidx.texcoord_index = (inidx.texcoord_index == -1) ? -1 : offset;
                outattrib.vertices.push_back(inattrib.vertices[3*inidx.vertex_index  ]);
                outattrib.vertices.push_back(inattrib.vertices[3*inidx.vertex_index+1]);
                outattrib.vertices.push_back(inattrib.vertices[3*inidx.vertex_index+2]);
                outattrib.normals.push_back(0.0f);
                outattrib.normals.push_back(0.0f);
                outattrib.normals.push_back(0.0f);
                if (inidx.texcoord_index != -1)
                {
                    outattrib.texcoords.push_back(inattrib.texcoords[2*inidx.texcoord_index  ]);
                    outattrib.texcoords.push_back(inattrib.texcoords[2*inidx.texcoord_index+1]);
                }
                remap[inidx.vertex_index] = offset;
            }
            outshape.mesh.indices.push_back(outidx);
        }
    }
}

static void computeSmoothingShapes(tinyobj::attrib_t &inattrib,
    std::vector<tinyobj::shape_t>& inshapes,
    std::vector<tinyobj::shape_t>& outshapes,
    tinyobj::attrib_t& outattrib)
{
    for (size_t s = 0, slen = inshapes.size() ; s < slen; ++s)
    {
        tinyobj::shape_t& inshape = inshapes[s];

        unsigned int numfaces = static_cast<unsigned int>(inshape.mesh.smoothing_group_ids.size());
        assert(numfaces);
        std::vector<std::pair<unsigned int,unsigned int>> sortedids(numfaces);
        for (unsigned int i = 0; i < numfaces; ++i)
        sortedids[i] = std::make_pair(inshape.mesh.smoothing_group_ids[i], i);
        sort(sortedids.begin(), sortedids.end());

        unsigned int activeid = sortedids[0].first;
        unsigned int id = activeid, idbegin = 0, idend = 0;
        // Faces are now bundled by smoothing group id, create shapes from these.
        while (idbegin < numfaces)
        {
            while (activeid == id && ++idend < numfaces) { id = sortedids[idend].first; }
            computeSmoothingShape(inattrib, inshape, sortedids, idbegin, idend, outshapes, outattrib);
            activeid = id;
            idbegin = idend;
        }
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

    auto attrib = reader.GetAttrib();
    auto shapes = reader.GetShapes();
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

            // h_normals.push_back(indices[3 * index + 0].normal_index);
            // h_normals.push_back(indices[3 * index + 1].normal_index);
            // h_normals.push_back(indices[3 * index + 2].normal_index);
        }
    }

    tinyobj::attrib_t outattrib;
    std::vector<tinyobj::shape_t> outshapes;
    computeSmoothingShapes(attrib, shapes, outshapes, outattrib);
    computeAllSmoothingNormals(outattrib, outshapes);

    // Check that the collected data is of the expected size
    uint32_t vertex_count = h_vertices.size();
    assert(3 * vertex_count == attrib.vertices.size());
    printf("%lu %lu\n", h_normals.size(), attrib.normals.size());
    //assert(h_normals.size() == attrib.normals.size());
    assert(h_triangles.size() == 3 * shapes[0].mesh.num_face_vertices.size());

    ViewerOptions options;
    options.window.size      = {1920,1080}; // Starting window size
    options.background_color = {.5f, .5f, .5f, 1.f};
    EngineHandle engine = nullptr;
    createEngine(options, &engine);

    float3 *d_coords   = nullptr;
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

    int block_size = 128;
    int grid_size    = (vertex_count + block_size - 1) / block_size;
    int displacement = 14;
    int displ_sign = 1;

    displayAsync(engine);
    while (isRunning(engine))
    {
        //std::cin.get();
        prepareViews(engine);
        // Perform 2D box filter on image using CUDA
        breatheKernel<<< grid_size, block_size >>>(d_coords, vertex_count);
        varySigma(displacement, displ_sign);
        updateViews(engine);
    }

    checkCuda(cudaFree(d_coords));
    checkCuda(cudaFree(d_triangles));
    destroyEngine(engine);

    return EXIT_SUCCESS;
}
