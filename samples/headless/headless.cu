// Headless offscreen rendering sample.
//
// Renders a static 3D point cloud to an image file with no on-screen window, exercising
// RenderMode::Headless. This is the foundation for remote rendering: the same render path
// runs on a GPU server with no display, and the produced frame is read back (here, to PPM;
// later, handed to a video encoder).
//
// Run from build/samples/:  ./run_headless [out.ppm] [width] [height] [point_count]

#include <curand_kernel.h>
#include <string> // std::stoul

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

// Generate uniformly distributed points in the unit cube
__global__ void initPos(float *coords, size_t point_count, unsigned int seed)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    curandState state;
    curand_init(seed, tidx, 0, &state);
    for (auto i = tidx; i < point_count; i += stride)
    {
        points[i] = { curand_uniform(&state), curand_uniform(&state), curand_uniform(&state) };
    }
}

int main(int argc, char *argv[])
{
    // Parameters
    std::string out_path     = "headless.ppm";
    int width                = 1920;
    int height               = 1080;
    unsigned int point_count = 10000;
    if (argc >= 2) out_path    = argv[1];
    if (argc >= 4) { width = std::stoi(argv[2]); height = std::stoi(argv[3]); }
    if (argc >= 5) point_count = std::stoul(argv[4]);

    checkCuda(cudaSetDevice(0));

    // Create a headless engine instance (no window/surface)
    ViewerOptions options;
    options.render_mode = RenderMode::Headless;
    options.window.size = { width, height };
    options.background_color = { 0.1f, 0.1f, 0.12f, 1.f };
    options.light_pos = { 0.f, 0.f, 10.f };
    InstanceHandle instance = nullptr;
    createInstance(options, &instance);

    // Allocate interop memory for point positions and fill it with a kernel
    float *d_coords = nullptr;
    AllocHandle points;
    allocLinear(instance, (void**)&d_coords, sizeof(float3) * point_count, &points);

    const unsigned int block_size = 256;
    const unsigned int grid_size = (point_count + block_size - 1) / block_size;
    initPos<<<grid_size, block_size>>>(d_coords, point_count, 12345u);
    checkCuda(cudaDeviceSynchronize());

    // Describe a 3D markers view over the point positions
    ViewDescription desc;
    desc.layout = Layout::make(point_count);
    desc.domain = DomainType::Domain3D;
    desc.type   = ViewType::Markers;
    desc.attributes[AttributeType::Position] = {
        .source = points,
        .size   = point_count,
        .format = FormatDescription::make<float3>(),
    };
    desc.default_size = 0.01f;
    desc.linewidth = 0.f;
    ViewHandle view = nullptr;
    createView(instance, &desc, &view);

    setCameraPosition(instance, {-.5f, -.5f, -3.f});

    // Render a single frame with no window and save it to disk
    renderHeadless(instance, []{}, 1);
    saveFrame(instance, out_path.c_str());

    destroyInstance(instance);
    checkCuda(cudaFree(d_coords));
    printf("Wrote headless frame to %s\n", out_path.c_str());
    return EXIT_SUCCESS;
}
