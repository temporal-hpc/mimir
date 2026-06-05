// Remote rendering server sample (step 2).
//
// Renders a live 3D point cloud headless (no window) and streams the raw frames over TCP to a
// connected client, applying the camera/pause control the client sends back. Pair with
// run_remote_client.
//
// Run from build/samples/:  ./run_remote_server [port] [width] [height] [point_count] [h264]
// Pass a non-zero final argument to request H.264 encoding (needs a build with ffmpeg support).

#include <curand_kernel.h>
#include <string> // std::stoul

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

__global__ void initRng(curandState *states, unsigned int count, unsigned int seed)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < count) { curand_init(seed, tidx, 0, &states[tidx]); }
}

__global__ void initPos(float *coords, size_t point_count, curandState *rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state = rng[tidx];
    for (auto i = tidx; i < point_count; i += stride)
    {
        points[i] = { curand_uniform(&state), curand_uniform(&state), curand_uniform(&state) };
    }
    rng[tidx] = state;
}

__global__ void integrate(float *coords, size_t point_count, curandState *rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state = rng[tidx];
    for (auto i = tidx; i < point_count; i += stride)
    {
        auto p = points[i];
        p.x += 0.05f * curand_normal(&state);
        p.y += 0.05f * curand_normal(&state);
        p.z += 0.05f * curand_normal(&state);
        p.x = fminf(fmaxf(p.x, 0.f), 1.f);
        p.y = fminf(fmaxf(p.y, 0.f), 1.f);
        p.z = fminf(fmaxf(p.z, 0.f), 1.f);
        points[i] = p;
    }
    rng[tidx] = state;
}

int main(int argc, char *argv[])
{
    unsigned short port      = 9000;
    int width                = 1280;
    int height               = 720;
    unsigned int point_count = 10000;
    bool use_h264            = false;
    if (argc >= 2) port        = static_cast<unsigned short>(std::stoi(argv[1]));
    if (argc >= 4) { width = std::stoi(argv[2]); height = std::stoi(argv[3]); }
    if (argc >= 5) point_count = std::stoul(argv[4]);
    if (argc >= 6) use_h264    = std::stoi(argv[5]) != 0;

    checkCuda(cudaSetDevice(0));

    const unsigned int block_size = 256;
    const unsigned int grid_size  = (point_count + block_size - 1) / block_size;
    const unsigned int rng_count  = grid_size * block_size;

    ViewerOptions options;
    options.render_mode = RenderMode::Headless;
    options.window.size = { width, height };
    options.background_color = { 0.1f, 0.1f, 0.12f, 1.f };
    options.light_pos = { 0.f, 0.f, 10.f };
    InstanceHandle instance = nullptr;
    createInstance(options, &instance);

    float *d_coords       = nullptr;
    curandState *d_states = nullptr;
    AllocHandle points;
    allocLinear(instance, (void**)&d_coords, sizeof(float3) * point_count, &points);
    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * rng_count));

    initRng<<<grid_size, block_size>>>(d_states, rng_count, 12345u);
    checkCuda(cudaDeviceSynchronize());
    initPos<<<grid_size, block_size>>>(d_coords, point_count, d_states);
    checkCuda(cudaDeviceSynchronize());

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

    printf("Serving frames on port %u (%dx%d, %u points). Connect with run_remote_client.\n",
        port, width, height, point_count);

    // Blocks until the client disconnects or sends Quit. The lambda advances the simulation
    // each (non-paused) frame over interop-mapped memory.
    serveRemote(instance, port, [&]{
        integrate<<<grid_size, block_size>>>(d_coords, point_count, d_states);
        checkCuda(cudaDeviceSynchronize());
    }, 0, use_h264);

    destroyInstance(instance);
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));
    return EXIT_SUCCESS;
}
