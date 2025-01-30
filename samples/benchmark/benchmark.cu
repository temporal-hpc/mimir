#include <cooperative_groups.h> // cooperative_groups::{sync, this_thread_block}
#include <string> // std::stoul

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda

#include "nvmlPower.hpp"

using namespace mimir;
namespace cg = cooperative_groups;

struct NBodyParams
{
    float time_step;
    float cluster_scale;
    float velocity_scale;
    float softening;
    float damping;
    float point_size;
    float x, y, z;

    void print()
    {
        printf("{ %f, %f, %f, %f, %f, %f, %f, %f, %f },\n", time_step,
            cluster_scale, velocity_scale, softening, damping, point_size, x, y, z
        );
    }
};

NBodyParams demoParams[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    {0.016f, 6.040f, 0.f, 1.f, 1.f, 0.760f, 0, 0, -50},
};

__constant__ float softeningSquared;

cudaError_t setSofteningSquared(float value)
{
    return cudaMemcpyToSymbol(softeningSquared, &value, sizeof(float), 0, cudaMemcpyHostToDevice);
}

struct SharedMemory {
    __device__ inline operator float4 *()
    {
        extern __shared__ int __smem[];
        return (float4 *)__smem;
    }

    __device__ inline operator const float4 *() const
    {
        extern __shared__ int __smem[];
        return (float4 *)__smem;
    }
};

struct DeviceData
{
    float4 *dPos[2];  // mapped host pointers
    float4 *dVel;
};

__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrtf(distSqr);
    float invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ float3 computeBodyAccel(float4 bodyPos, float4 *positions, int numTiles,
    cg::thread_block cta)
{
    float4 *sharedPos = SharedMemory();
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
        cg::sync(cta);

// This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128
        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);
        }
        cg::sync(cta);
    }

    return acc;
}

__global__ void integrateBodies(float4 *__restrict__ newPos, float4 *__restrict__ oldPos,
    float4 *vel, unsigned int deviceNumBodies, float deltaTime, float damping, int numTiles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies) { return; }

    float4 position = oldPos[index];
    float3 accel = computeBodyAccel(position, oldPos, numTiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in
    // bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    float4 velocity = vel[index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[index] = position;
    vel[index] = velocity;
}

void integrateNbodySystem(DeviceData deviceData, unsigned int currentRead, float deltaTime,
    float damping, unsigned int numBodies, int blockSize)
{
    int numBlocks = (numBodies + blockSize - 1) / blockSize;
    int numTiles = (numBodies + blockSize - 1) / blockSize;
    int sharedMemSize = blockSize * 4 * sizeof(float);  // 4 floats for pos

    integrateBodies<<<numBlocks, blockSize, sharedMemSize>>>(
        deviceData.dPos[1 - currentRead],
        deviceData.dPos[currentRead],
        deviceData.dVel,
        numBodies, deltaTime, damping, numTiles
    );

    // check if kernel invocation generated an error
    checkCuda(cudaGetLastError());
}

inline float dot(float3 v0, float3 v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

void randomizeBodies(float *pos, float *vel, float *color,
    float clusterScale, float velocityScale, int numBodies, bool vec4vel)
{
    float scale = clusterScale * std::max<float>(1.0f, numBodies / (1024.0f));
    float vscale = velocityScale * scale;

    int p = 0, v = 0;
    int i = 0;

    while (i < numBodies)
    {
        float3 point;
        // const int scale = 16;
        point.x = rand() / (float)RAND_MAX * 2 - 1;
        point.y = rand() / (float)RAND_MAX * 2 - 1;
        point.z = rand() / (float)RAND_MAX * 2 - 1;
        float lenSqr = dot(point, point);

        if (lenSqr > 1) continue;

        float3 velocity;
        velocity.x = rand() / (float)RAND_MAX * 2 - 1;
        velocity.y = rand() / (float)RAND_MAX * 2 - 1;
        velocity.z = rand() / (float)RAND_MAX * 2 - 1;
        lenSqr = dot(velocity, velocity);

        if (lenSqr > 1) continue;

        pos[p++] = point.x * scale;  // pos.x
        pos[p++] = point.y * scale;  // pos.y
        pos[p++] = point.z * scale;  // pos.z
        pos[p++] = 1.0f;             // mass

        vel[v++] = velocity.x * vscale;  // pos.x
        vel[v++] = velocity.y * vscale;  // pos.x
        vel[v++] = velocity.z * vscale;  // pos.x

        if (vec4vel) vel[v++] = 1.0f;  // inverse mass
        i++;
    }

    if (color != nullptr)
    {
        int v = 0;
        for (int i = 0; i < numBodies; i++)
        {
            // const int scale = 16;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = 1.0f;
        }
    }
}

int main(int argc, char *argv[])
{
    // CUDA initialization
    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));
    const unsigned int block_size = 256;

    // Default experiment parameters
    int width                = 1920;
    int height               = 1080;
    unsigned int body_count  = 4096;
    int iter_count           = 1000000;
    PresentMode present_mode = PresentMode::Immediate;
    int target_fps           = 0;
    bool enable_sync         = true;
    bool use_interop         = true;

    // Parse parameters from command line
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) body_count   = std::stoul(argv[3]);
    if (argc >= 5) iter_count   = std::stoi(argv[4]);
    if (argc >= 6) present_mode = static_cast<PresentMode>(std::stoi(argv[5]));
    if (argc >= 7) target_fps   = std::stoi(argv[6]);
    if (argc >= 8) enable_sync  = static_cast<bool>(std::stoi(argv[7]));
    if (argc >= 9) use_interop  = static_cast<bool>(std::stoi(argv[8]));

    // Determine execution mode for benchmarking and write CSV column names
    std::string mode;
    if (width == 0 && height == 0) mode = use_interop? "mimir" : "original";
    else mode = enable_sync? "sync" : "desync";

    bool display = true;
    if (width == 0 || height == 0)
    {
        width = height = 10;
        display = false;
    }

    ViewerOptions options;
    options.window.size   = {width,height}; // Starting window size
    options.show_metrics  = false; // Show metrics window in GUI
    options.report_period = 0; // Print relevant usage stats every N seconds
    options.present = {
        .mode        = present_mode,
        .enable_sync = enable_sync,
        .target_fps  = target_fps,
    };
    EngineHandle engine = nullptr;
    createEngine(options, &engine);

    auto nbody_memsize = sizeof(float4) * body_count;
    DeviceData data;
    checkCuda(cudaMalloc((void**)&data.dVel, nbody_memsize));

    mimir::ViewHandle views[2];
    if (use_interop)
    {
        mimir::AllocHandle allocs[2];
        allocLinear(engine, (void**)&data.dPos[0], nbody_memsize, &allocs[0]);
        allocLinear(engine, (void**)&data.dPos[1], nbody_memsize, &allocs[1]);

        ViewDescription desc{
        .element_count = body_count,
        .view_type     = ViewType::Markers,
        .domain_type   = DomainType::Domain3D,
        .extent        = {1, 1, 1},
        .attributes    = {
            { AttributeType::Position, {
                .source = allocs[0],
                .size   = body_count,
                .format = FormatDescription::make<float4>(),
                .indices = {},
                .index_size = 0,
            }}}
        };
        createView(engine, &desc, &views[0]);
        views[0]->default_size = 100.f;

        desc.attributes[AttributeType::Position].source = allocs[1];
        createView(engine, &desc, &views[1]);
        views[1]->default_size = 100.f;
        views[1]->visible = false;
    }
    else // Run the simulation without display
    {
        checkCuda(cudaMalloc((void**)&data.dPos[0], nbody_memsize));
        checkCuda(cudaMalloc((void**)&data.dPos[1], nbody_memsize));
    }

    // Initialize simulation
    unsigned int current_read  = 0;
    unsigned int current_write = 1;
    float delta_time = 0.001f;

    NBodyParams params = demoParams[0];
    setSofteningSquared(params.softening);
    float *h_pos = new float[body_count * 4];
    float *h_vel = new float[body_count * 4];
    randomizeBodies(h_pos, h_vel, nullptr, params.cluster_scale, params.velocity_scale, body_count, true);
    checkCuda(cudaMemcpy(data.dPos[current_read], h_pos, nbody_memsize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(data.dVel, h_vel, nbody_memsize, cudaMemcpyHostToDevice));
    delete[] h_pos;
    delete[] h_vel;

    // Start display and measurements
    GPUPowerBegin("gpu", 100);
    if (display) displayAsync(engine);

    // Main simulation loop
    for (int i = 0; i < iter_count; ++i)
    {
        if (display) prepareViews(engine);
        integrateNbodySystem(data, current_read, delta_time, params.damping, body_count, block_size);
        std::swap(current_read, current_write);
        if (display)
        {
            views[0]->toggleVisibility();
            views[1]->toggleVisibility();
            updateViews(engine);
        }
    }

    // Retrieve metrics
    printf("%s,%u,", mode.c_str(), body_count);
    getMetrics(engine);

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

    // Cleanup
    exit(engine);
    destroyEngine(engine);
    checkCuda(cudaFree(data.dPos[0]));
    checkCuda(cudaFree(data.dPos[1]));
    checkCuda(cudaFree(data.dVel));

    return EXIT_SUCCESS;
}
