#include <string> // std::stoul

#include "nbody_gpu.cuh"
#include "nbody_cpu.hpp"
#include "nvmlPower.hpp"

#include <mimir/mimir.hpp>
using namespace mimir;

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

NBodyParams demo_params[] = {
    {0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    {0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    {0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    {0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    {0.016f, 6.040f, 0.f, 1.f, 1.f, 0.760f, 0, 0, -50},
};

struct BenchmarkInput
{
    int width;
    int height;
    uint32_t body_count;
    int iter_count;
    PresentMode present;
    int target_fps;
    bool enable_sync;
    bool display;
    bool compute_gpu;

    // Default experiment parameters
    static BenchmarkInput defaultValues()
    {
        return BenchmarkInput{
            .width       = 1920,
            .height      = 1080,
            .body_count  = 77824,
            .iter_count  = 1000000,
            .present     = PresentMode::Immediate,
            .target_fps  = 0,
            .enable_sync = true,
            .display     = true,
            .compute_gpu = true,
        };
    }
};

struct BenchmarkResults
{
    float framerate;
    float compute_time;
    float pipeline_time;
    float graphics_time;
    float freemem;
    float usedmem;
    float gpu_mem_usage;
    float gpu_mem_budget;
};

void formatResults(BenchmarkInput input, BenchmarkResults output)
{
    // Determine execution mode for benchmarking and write CSV column names
    std::string mode;
    if (input.width == 0 && input.height == 0)
    {
        mode = input.display? "mimir" : "no_disp";
    }
    else { mode = input.enable_sync? "sync" : "desync"; }

    std::string resolution = "None";
    if      (input.width == 1920 && input.height == 1080) { resolution = "FHD"; }
    else if (input.width == 2560 && input.height == 1440) { resolution = "QHD"; }
    else if (input.width == 3840 && input.height == 2160) { resolution = "UHD"; }

    printf("%s,%s,%d,%d,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(),
        resolution.c_str(),
        input.body_count,
        input.target_fps,
        output.framerate,
        output.compute_time,
        output.pipeline_time,
        output.graphics_time,
        output.gpu_mem_usage,
        output.gpu_mem_budget
    );
}

BenchmarkResults runExperiment(BenchmarkInput input, NBodyParams params)
{
    // CUDA initialization
    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));
    // Kernel block size
    constexpr unsigned int block_size = 256;

    if (input.display == false) { input.width = input.height = 1; }

    ViewerOptions options;
    options.window.size = {input.width, input.height}; // Starting window size
    options.background_color = {0.f, 0.f, 0.f, 1.f};
    options.present = {
        .mode        = input.present,
        .enable_sync = input.enable_sync,
        .target_fps  = input.target_fps,
    };
    EngineHandle engine = nullptr;
    createEngine(options, &engine);
    setCameraPosition(engine, {params.x, params.y, params.z});

    auto nbody_memsize = sizeof(float4) * input.body_count;
    DeviceData data;
    checkCuda(cudaMalloc((void**)&data.dVel, nbody_memsize));

    mimir::ViewHandle views[2];
    if (input.display)
    {
        mimir::AllocHandle allocs[2];
        allocLinear(engine, (void**)&data.dPos[0], nbody_memsize, &allocs[0]);
        allocLinear(engine, (void**)&data.dPos[1], nbody_memsize, &allocs[1]);

        ViewDescription desc{
            .layout      = Layout::make(input.body_count),
            .view_type   = ViewType::Markers,
            .domain_type = DomainType::Domain3D,
            .attributes  = {
                { AttributeType::Position, {
                    .source = allocs[0],
                    .size   = input.body_count,
                    .format = FormatDescription::make<float4>(),
                }}
            },
            .visible       = true,
            .default_color = {1.f, 1.f, 1.f, 1.f},
            .default_size  = params.point_size / 5.f,
            .scale         = {1.f, 1.f, 1.f},
        };
        createView(engine, &desc, &views[0]);

        desc.visible = false;
        desc.attributes[AttributeType::Position].source = allocs[1];
        createView(engine, &desc, &views[1]);
    }
    else // Run the simulation without display
    {
        checkCuda(cudaMalloc((void**)&data.dPos[0], nbody_memsize));
        checkCuda(cudaMalloc((void**)&data.dPos[1], nbody_memsize));
    }

    // Initialize simulation
    unsigned int current_read  = 0;
    unsigned int current_write = 1;

    NBodyConfig config = NBodyConfig::Shell;
    setSofteningSquared(params.softening);
    float *h_pos = new float[input.body_count * 4];
    float *h_vel = new float[input.body_count * 4];
    randomizeBodies(config, h_pos, h_vel, nullptr,
        params.cluster_scale, params.velocity_scale, input.body_count, true
    );
    checkCuda(cudaMemcpy(data.dPos[current_read], h_pos, nbody_memsize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(data.dVel, h_vel, nbody_memsize, cudaMemcpyHostToDevice));
    delete[] h_pos;
    delete[] h_vel;

    // Start display and measurements
    setCameraPosition(engine, {1.f, 1.f, -3.f});
    GPUPowerBegin("gpu", 100);
    if (input.display) displayAsync(engine);

    // Main simulation loop
    for (int i = 0; i < input.iter_count; ++i)
    {
        if (input.display) prepareViews(engine);
        integrateNbodySystem(data, current_read, params.time_step,
            params.damping, input.body_count, block_size
        );
        std::swap(current_read, current_write);
        if (input.display)
        {
            toggleVisibility(views[0]);
            toggleVisibility(views[1]);
            updateViews(engine);
        }
    }

    // Retrieve metrics
    //auto metrics = getMetrics(engine);
    BenchmarkResults output;

    // Nvml memory report
    {
        nvmlMemory_v2_t meminfo;
        meminfo.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);

        constexpr double gigabyte = 1024.0 * 1024.0 * 1024.0;
        output.freemem  = meminfo.free / gigabyte;
        //output.reserved = meminfo.reserved / gigabyte;
        //output.totalmem = meminfo.total / gigabyte;
        output.usedmem  = meminfo.used / gigabyte;
    }
    GPUPowerEnd();

    // Cleanup
    exit(engine);
    destroyEngine(engine);
    checkCuda(cudaFree(data.dPos[0]));
    checkCuda(cudaFree(data.dPos[1]));
    checkCuda(cudaFree(data.dVel));

    return output;
}

int main(int argc, char *argv[])
{
    auto input = BenchmarkInput::defaultValues();
    NBodyParams params = demo_params[3];

    // Parse parameters from command line
    if (argc >= 3) { input.width = std::stoi(argv[1]); input.height = std::stoi(argv[2]); }
    if (argc >= 4) input.body_count   = std::stoul(argv[3]);
    if (argc >= 5) input.iter_count   = std::stoi(argv[4]);
    if (argc >= 6) input.present      = static_cast<PresentMode>(std::stoi(argv[5]));
    if (argc >= 7) input.target_fps   = std::stoi(argv[6]);
    if (argc >= 8) input.enable_sync  = static_cast<bool>(std::stoi(argv[7]));
    if (argc >= 9) input.display      = static_cast<bool>(std::stoi(argv[8]));
    if (argc >= 10) input.compute_gpu = static_cast<bool>(std::stoi(argv[9]));

    auto result = runExperiment(input, params);
    formatResults(input, result);

    return EXIT_SUCCESS;
}
