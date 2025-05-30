#include <curand_kernel.h>
#include <iostream> // std::cout
#include <fstream> // std::ofstream
#include <string> // std::stoul

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

#include "nvmlPower.hpp"

// Init RNG states
__global__ void initRng(curandState *states, unsigned int rng_count, unsigned int seed)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tidx, 0, &states[tidx]);
}

// Init starting positions
__global__ void initPos(float *coords, size_t point_count, curandState *rng, uint3 extent)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state = rng[tidx];
    for (int i = tidx; i < point_count; i += stride)
    {
        auto rx = extent.x * curand_uniform(&state);
        auto ry = extent.y * curand_uniform(&state);
        auto rz = extent.z * curand_uniform(&state);
        points[i] = {rx, ry, rz};
    }
    rng[tidx] = state;
}

__global__ void integrate3d(float *coords, size_t point_count, curandState *rng, uint3 extent)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto local_state = rng[tidx];
    for (int i = tidx; i < point_count; i += stride)
    {
        // Read current position
        auto p = points[i];

        // Generate random displacements with device RNG state
        p.x += curand_normal(&local_state);
        p.y += curand_normal(&local_state);
        p.z += curand_normal(&local_state);

        // Correct positions to bounds
        if (p.x > extent.x) p.x = extent.x;
        if (p.x < 0) p.x = 0;
        if (p.y > extent.x) p.y = extent.y;
        if (p.y < 0) p.y = 0;
        if (p.z > extent.z) p.z = extent.z;
        if (p.z < 0) p.z = 0;

        // Write updated position
        points[i] = p;
    }
    rng[tidx] = local_state;
}

int main(int argc, char *argv[])
{
    // Set manually CUDA device to 0; change if needed
    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));

    // Retrieve number of streaming multiprocessors (SMs)
    int sm_count = -1;
    checkCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

    // Retrieve max number of threads per SM
    int max_sm_thread_count = -1;
    checkCuda(cudaDeviceGetAttribute(
        &max_sm_thread_count, cudaDevAttrMaxThreadsPerMultiProcessor, device_id)
    );

    // Determine kernel parameters from previous values
    const unsigned int block_size = 256;
    const int max_block_count = max_sm_thread_count / block_size;
    const int rng_state_count = max_block_count * sm_count * block_size;
    const int grid_size = (rng_state_count + block_size - 1) / block_size;

    // Experiment constants
    const unsigned int seed = 12345u;
    const uint3 extent      = {200, 200, 200};

    // Default experiment parameters
    int width                = 1920;
    int height               = 1080;
    unsigned int point_count = 100;
    int iter_count           = 10000;
    PresentMode present_mode = PresentMode::Immediate;
    int target_fps           = 0;
    bool enable_sync         = true;
    bool use_interop         = true;

    //printf("%d %d %d\n", sm_count, max_block_count, grid_size);

    // Parse parameters from command line
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) point_count  = std::stoul(argv[3]);
    if (argc >= 5) iter_count   = std::stoi(argv[4]);
    if (argc >= 6) present_mode = static_cast<PresentMode>(std::stoi(argv[5]));
    if (argc >= 7) target_fps   = std::stoi(argv[6]);
    if (argc >= 8) enable_sync  = static_cast<bool>(std::stoi(argv[7]));
    if (argc >= 9) use_interop  = static_cast<bool>(std::stoi(argv[8]));

    // Determine execution mode for benchmarking and write CSV column names
    std::string mode;
    if (width == 0 && height == 0) mode = use_interop? "interop" : "cudaMalloc";
    else mode = enable_sync? "sync" : "desync";

    bool display = true;
    if (width == 0 || height == 0)
    {
        width = height = 10;
        display = false;
    }

    ViewerOptions options;
    options.window.size = {width,height}; // Starting window size
    options.present = {
        .mode        = present_mode,
        .enable_sync = enable_sync,
        .target_fps  = target_fps,
    };
    InstanceHandle instance = nullptr;
    createInstance(options, &instance);

    float *d_coords       = nullptr;
    curandState *d_states = nullptr;
    if (use_interop)
    {
        AllocHandle points;
        allocLinear(instance, (void**)&d_coords, sizeof(float3) * point_count, &points);

        ViewHandle view = nullptr;
        ViewDescription desc;
        desc.layout      = Layout::make(point_count);
        desc.domain = DomainType::Domain3D;
        desc.type   = ViewType::Markers;
        desc.attributes[AttributeType::Position] = {
            .source = points,
            .size   = point_count,
            .format = FormatDescription::make<float3>(),
        };
        desc.default_size = 0.1f;
        desc.scale = { 0.01f, 0.01f, 0.01f };
        createView(instance, &desc, &view);
    }
    else // Run the simulation without display
    {
        checkCuda(cudaMalloc((void**)&d_coords, sizeof(float3) * point_count));
    }
    setCameraPosition(instance, {-2.f, -2.f, -7.f});

    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * rng_state_count));
    initRng<<<grid_size, block_size>>>(d_states, rng_state_count, seed);
    checkCuda(cudaDeviceSynchronize());
    initPos<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
    checkCuda(cudaDeviceSynchronize());

    GPUPowerBegin("gpu", 100);
    if (display) displayAsync(instance);
    for (int i = 0; i < iter_count && isRunning(instance); ++i)
    {
        if (display) prepareViews(instance);
        integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
        if (display) updateViews(instance);
    }
    printf("%s,%u,", mode.c_str(), point_count);
    getMetrics(instance);

    // Nvml memory report
    {
        nvmlMemory_v2_t meminfo;
        meminfo.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);

        constexpr double gigabyte = 1024.0 * 1024.0 * 1024.0;
        double freemem = meminfo.free / gigabyte;
        double reserved = meminfo.reserved / gigabyte;
        double totalmem = meminfo.total / gigabyte;
        double usedmem = meminfo.used / gigabyte;
        printf("%lf,%lf,", freemem, usedmem);
    }

    GPUPowerEnd();

    destroyInstance(instance);
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
