#include <chrono>
#include <random>
#include <string> // std::stoul, std::string (HUD text)
#include <vector>

#include "benchmark.hpp"
#include "nbody_gpu.cuh"
#include "nbody_cpu.hpp"
#include "nvmlPower.hpp"

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

struct GPUMemoryMetrics {
    double free;
    double reserved;
    double total;
    double used;
};

struct BenchmarkResult {
    PerformanceMetrics perf;
    GPUPowerMetrics power;
    GPUMemoryMetrics memory;
    int iters;  // iterations actually executed (divide the time TOTALS by this for per-frame averages)
};

static std::string sf(float v)  { char b[32]; snprintf(b, sizeof(b), "%f", v); return b; }
static std::string sd(int v)    { return std::to_string(v); }
static std::string smb(size_t bytes) {
    char b[32]; snprintf(b, sizeof(b), "%.2f MB", bytes / (1024.0 * 1024.0)); return b;
}

static void printSystemInfo(size_t nbody_memsize, bool display)
{
    char gpu_name[256] = "Unknown";
    nvmlDeviceGetName(getNvmlDevice(), gpu_name, sizeof(gpu_name));

    nvmlMemory_v2_t mi;
    mi.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
    nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
    constexpr double gb = 1024.0 * 1024.0 * 1024.0;

    const char *kind = display ? "interop" : "CUDA";
    fprintf(stderr, "GPU: %s\n", gpu_name);
    fprintf(stderr, "Total GPU memory: %.2f GB\n", mi.total / gb);
    fprintf(stderr, "Buffers:\n");
    fprintf(stderr, "  dPos[0]  (%s):    %s\n", kind, smb(nbody_memsize).c_str());
    fprintf(stderr, "  dPos[1]  (%s):    %s\n", kind, smb(nbody_memsize).c_str());
    fprintf(stderr, "  dVel     (CUDA):     %s\n", smb(nbody_memsize).c_str());
    fprintf(stderr, "  Total:               %s\n", smb(3 * nbody_memsize).c_str());
}

static void printAligned(std::initializer_list<std::pair<const char*, std::string>> cols)
{
    std::vector<int> w;
    for (auto& [h, v] : cols)
        w.push_back((int)(strlen(h) > v.size() ? strlen(h) : v.size()));
    int i = 0;
    for (auto& [h, v] : cols) fprintf(stderr, "%-*s  ", w[i++], h);
    fprintf(stderr, "\n");
    i = 0;
    for (auto& [h, v] : cols) fprintf(stderr, "%-*s  ", w[i++], v.c_str());
    fprintf(stderr, "\n");
}

void formatResults(BenchmarkInput input, BenchmarkResult result)
{
    // Determine execution mode for benchmarking and write CSV column names
    std::string mode;
    if (input.width == 0 && input.height == 0)
    {
        mode = input.display? "mimir" : "none";
    }
    else { mode = input.enable_interop_sync? "sync" : "desync"; }

    std::string resolution = "None";
    if      (input.width == 1920 && input.height == 1080) { resolution = "FHD"; }
    else if (input.width == 2560 && input.height == 1440) { resolution = "QHD"; }
    else if (input.width == 3840 && input.height == 2160) { resolution = "UHD"; }

    auto library = result.perf;
    auto gpu = result.power;
    auto nvml = result.memory;

    // pack_time/d2h_time/h2h_time are 0 for mimir (zero-copy, no pack step).
    // Column layout matches nbody-datoviz for direct CSV comparison.
    // Column names carry units; time columns are TOTALS over the run in seconds (matching
    // nbody-datoviz), memory in GB, power in W, energy in J.
    printAligned({
        {"mode",             mode},
        {"windowres",        resolution},
        {"N",                sd(input.body_count)},
        {"iters",            sd(result.iters)},
        {"framerate_fps",    sf(library.frame_rate)},
        {"compute_time_s",   sf(library.times.compute)},
        {"pipeline_time_s",  sf(library.times.pipeline)},
        {"graphics_time_s",  sf(library.times.graphics)},
        {"vk_usage_gb",      sf(library.devmem.usage)},
        {"vk_budget_gb",     sf(library.devmem.budget)},
        {"gpu_power_w",      sf(gpu.average_power)},
        {"gpu_energy_j",     sf(gpu.total_energy)},
        {"gpu_time_s",       sf(gpu.total_time)},
        {"nvml_free_gb",     sf(nvml.free)},
        {"nvml_reserved_gb", sf(nvml.reserved)},
        {"nvml_total_gb",    sf(nvml.total)},
        {"nvml_used_gb",     sf(nvml.used)},
        {"pack_time_s",      sf(0.f)},
        {"d2h_time_s",       sf(0.f)},
        {"h2h_time_s",       sf(0.f)},
    });
    printf("%s,%s,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(),
        resolution.c_str(),
        input.body_count,
        result.iters,
        library.frame_rate,
        library.times.compute,
        library.times.pipeline,
        library.times.graphics,
        library.devmem.usage,
        library.devmem.budget,
        gpu.average_power,
        gpu.total_energy,
        gpu.total_time,
        nvml.free,
        nvml.reserved,
        nvml.total,
        nvml.used,
        0.f, 0.f, 0.f
    );
}


inline float normalize(float3 &vector)
{
    float dist = sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    if (dist > 1e-6)
    {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}

inline float dot(float3 v0, float3 v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

inline float3 cross(float3 v0, float3 v1)
{
    float3 rt;
    rt.x = v0.y * v1.z - v0.z * v1.y;
    rt.y = v0.z * v1.x - v0.x * v1.z;
    rt.z = v0.x * v1.y - v0.y * v1.x;
    return rt;
}

void randomizeBodies(NBodyConfig config, float *pos, float *vel, float *color,
    float cluster_scale, float velocity_scale, int body_count, bool vec4vel)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rand_pos(-1.f, 1.f);
    std::uniform_real_distribution<float> rand_unit(0.f, 1.f);

    //float weight = 100000.f;
    float mass = 1.f;//weight / body_count;
    float inv_mass = 1.f;//body_count / weight;
    switch (config)
    {
        default:
        case NBodyConfig::Random:
        {
            float scale = cluster_scale * std::max<float>(1.0f, body_count / (1024.0f));
            float vscale = velocity_scale * scale;

            int p = 0, v = 0, i = 0;
            while (i < body_count)
            {
                float3 point;
                // const int scale = 16;
                point.x = rand_pos(rng);
                point.y = rand_pos(rng);
                point.z = rand_pos(rng);
                float len_sqr = dot(point, point);

                if (len_sqr > 1) continue;

                float3 velocity;
                velocity.x = rand_pos(rng);
                velocity.y = rand_pos(rng);
                velocity.z = rand_pos(rng);
                len_sqr = dot(velocity, velocity);

                if (len_sqr > 1) continue;

                pos[p++] = point.x * scale;  // pos.x
                pos[p++] = point.y * scale;  // pos.y
                pos[p++] = point.z * scale;  // pos.z
                pos[p++] = mass;             // mass

                vel[v++] = velocity.x * vscale;  // pos.x
                vel[v++] = velocity.y * vscale;  // pos.x
                vel[v++] = velocity.z * vscale;  // pos.x

                if (vec4vel) vel[v++] = inv_mass;  // inverse mass

                i++;
            }
        } break;

        case NBodyConfig::Shell:
        {
            float scale = cluster_scale;
            float vscale = scale * velocity_scale;
            float inner = 2.5f * scale;
            float outer = 4.0f * scale;

            int p = 0, v = 0, i = 0;
            while (i < body_count)
            {
                float x, y, z;
                x = rand_pos(rng);
                y = rand_pos(rng);
                z = rand_pos(rng);

                float3 point = {x, y, z};
                float len = normalize(point);
                if (len > 1) { continue; }

                pos[p++] = point.x * (inner + (outer - inner) * rand_unit(rng));
                pos[p++] = point.y * (inner + (outer - inner) * rand_unit(rng));
                pos[p++] = point.z * (inner + (outer - inner) * rand_unit(rng));
                pos[p++] = mass;

                x = 0.0f;  // * (rand() / (float) RAND_MAX * 2 - 1);
                y = 0.0f;  // * (rand() / (float) RAND_MAX * 2 - 1);
                z = 1.0f;  // * (rand() / (float) RAND_MAX * 2 - 1);
                float3 axis = {x, y, z};
                normalize(axis);

                if (1 - dot(point, axis) < 1e-6)
                {
                    axis.x = point.y;
                    axis.y = point.x;
                    normalize(axis);
                }

                // if (point.y < 0) axis = scalevec(axis, -1);
                float3 vv = {pos[4 * i], pos[4 * i + 1], pos[4 * i + 2]};
                vv = cross(vv, axis);
                vel[v++] = vv.x * vscale;
                vel[v++] = vv.y * vscale;
                vel[v++] = vv.z * vscale;

                if (vec4vel) { vel[v++] = inv_mass; }

                i++;
            }
        } break;

        case NBodyConfig::Expand:
        {
            float scale = cluster_scale * body_count / (1024.f);

            if (scale < 1.0f) { scale = cluster_scale; }
            float vscale = scale * velocity_scale;
            int p = 0, v = 0;

            for (int i = 0; i < body_count;)
            {
                float3 point;
                point.x = rand_pos(rng);
                point.y = rand_pos(rng);
                point.z = rand_pos(rng);

                float len_sqr = dot(point, point);
                if (len_sqr > 1) { continue; }

                pos[p++] = point.x * scale;   // pos.x
                pos[p++] = point.y * scale;   // pos.y
                pos[p++] = point.z * scale;   // pos.z
                pos[p++] = mass;              // mass
                vel[v++] = point.x * vscale;  // pos.x
                vel[v++] = point.y * vscale;  // pos.x
                vel[v++] = point.z * vscale;  // pos.x

                if (vec4vel) vel[v++] = inv_mass;  // inverse mass

                i++;
            }
        } break;
    }

    if (color != nullptr)
    {
        std::uniform_real_distribution<float> rand_color(0, 1);
        int v = 0;
        for (int i = 0; i < body_count; i++)
        {
            color[v++] = rand_color(rng);
            color[v++] = rand_color(rng);
            color[v++] = rand_color(rng);
            color[v++] = 1.0f;
        }
    }
}

struct HudData {
    unsigned int n             = 0;
    int          frame         = 0;
    float        fps           = 0.f;
    float        compute_ms    = 0.f;
    float        render_ms     = 0.f;
    float        wait_ms       = 0.f;  // CPU blocked on fence + swapchain acquire (backpressure)
    float        record_ms     = 0.f;  // CPU command-buffer recording
    float        submit_ms     = 0.f;  // CPU vkQueueSubmit + vkQueuePresentKHR
    float        gpu_ms        = 0.f;  // true end-to-end GPU frame latency (submit -> fence)
    float        gpu_watts     = 0.f;
    char         gpu_name[256] = {};
    char         gpu_device[64] = {};  // "N (CC major.minor)"
    float        gpu_total_gb  = 0.f;  // total VRAM (NVML)
    // VRAM used (NVML) dismembered so the sub-parts sum to it. External/CUDA are anchored on
    // measured NVML checkpoints at startup; Render/Vulkan are computed; Mimir is the remainder.
    float        vram_used_mb     = 0.f;
    float        vram_external_mb = 0.f;  // measured: NVML used before we touch the GPU
    float        cuda_ctx_mb      = 0.f;  // measured: CUDA context reservation (cudaFree(0))
    float        buf_total_mb     = 0.f;  // computed: our CUDA sim device buffers (dPos x2 + dVel)
    float        render_mb        = 0.f;  // computed: extra render geometry (0: point-sprite markers)
    float        vulkan_mb        = 0.f;  // computed: Vulkan render targets (swapchain + depth)
};

// Format the live HUD as plain text for mimir's built-in overlay (setHudText). All numbers are
// collected in plain C++/CUDA/NVML above; this is pure string formatting -- no ImGui, no GUI code.
static std::string formatHud(const HudData& h)
{
    const float mimir_mb = h.vram_used_mb - h.vram_external_mb - h.cuda_ctx_mb
        - h.buf_total_mb - h.render_mb - h.vulkan_mb;
    char b[1024];
    snprintf(b, sizeof(b),
        "GPU       %s\n"
        "Device    %s\n"
        "VRAM      %.1f GB\n"
        "VRAM used %.0f MB\n"
        "  External   %.0f MB\n"
        "  CUDA ctx   %.0f MB\n"
        "  CUDA bufs  %.1f MB\n"
        "  Render geo %.3f MB\n"
        "  Vulkan tgt %.0f MB\n"
        "  Mimir      %.0f MB\n"
        "\n"
        "Bodies    %u\n"
        "Frame     %d\n"
        "Compute   %.2f ms\n"
        "GPU frame %.2f ms\n"
        "  Wait     %.2f ms\n"
        "  Record   %.2f ms\n"
        "  Submit   %.2f ms\n"
        "Power     %.1f W",
        h.gpu_name, h.gpu_device, h.gpu_total_gb, h.vram_used_mb,
        h.vram_external_mb, h.cuda_ctx_mb, h.buf_total_mb, h.render_mb, h.vulkan_mb, mimir_mb,
        h.n, h.frame, h.compute_ms, h.gpu_ms, h.wait_ms, h.record_ms, h.submit_ms, h.gpu_watts);
    return b;
}

// Whole-GPU VRAM in use right now (NVML "used"), in MB. Lazily nvmlInit()s its own device handle
// so it can be sampled at startup checkpoints -- BEFORE GPUPowerBegin(), which starts the energy
// measurement window later and must not be moved. NVML refcounts init(), so this coexists with it.
static double sampleVramUsedMB()
{
    static nvmlDevice_t dev = nullptr;
    if (dev == nullptr)
    {
        if (nvmlInit() != NVML_SUCCESS) return 0.0;
        if (nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS) { dev = nullptr; return 0.0; }
    }
    nvmlMemory_v2_t mi;
    mi.version = (unsigned int)(sizeof(mi) | (2 << 24U));
    return (nvmlDeviceGetMemoryInfo_v2(dev, &mi) == NVML_SUCCESS)
             ? (double)mi.used / (1024.0 * 1024.0) : 0.0;
}

BenchmarkResult runExperiment(BenchmarkInput input, NBodyParams params)
{
    // VRAM checkpoint #1: before we touch the GPU -- baseline held by other processes.
    const double vram_external = sampleVramUsedMB();

    // CUDA initialization
    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));
    // VRAM checkpoint #2: force the primary CUDA context to exist (cudaFree(0)) and measure its
    // reservation on its own, before our buffers or the mimir/Vulkan renderer are created.
    checkCuda(cudaFree(0));
    double cuda_ctx_mb = sampleVramUsedMB() - vram_external;
    if (cuda_ctx_mb < 0.0) cuda_ctx_mb = 0.0;
    // Kernel block size
    constexpr unsigned int block_size = 256;

    if (input.display == false) { input.width = input.height = 1; }

    ViewerOptions options{};
    options.window.title = "Mimir - nbody";
    options.window.size = {input.width, input.height}; // Starting window size
    // Cheapest possible particles: LightModel::None draws markers as UNLIT native point
    // sprites (Flat2D: no geometry shader, no lighting, no custom depth) -- the same cost
    // class as datoviz's flat disc markers in samples/nbody-datoviz. The library default
    // (Phong) would render ray-traced sphere impostors instead.
    options.light_model = LightModel::None;
    options.background_color = {0.f, 0.f, 0.f, 1.f};
    options.present.mode = input.present;
    options.present.enable_interop_sync = input.enable_interop_sync;
    options.present.enable_fps_limit = false;  // always uncapped
    options.show_panel = false;  // Ctrl+G shows the scene panel; F1 toggles ALL GUI windows.
    // Built-in HUD overlay (F2). The benchmark pushes its metrics into it via setHudText each frame,
    // so this file links no ImGui and writes no GUI code -- measurement stays pure C++/CUDA/NVML.
    options.show_hud = true;

    InstanceHandle instance = nullptr;
    createInstance(options, &instance);
    setCameraPosition(instance, {params.x, params.y, params.z - 1.f});

    auto nbody_memsize = sizeof(float4) * input.body_count;
    DeviceData device;
    checkCuda(cudaMalloc((void**)&device.dVel, nbody_memsize));

    mimir::ViewHandle views[2];
    if (input.display)
    {
        mimir::AllocHandle allocs[2];
        allocLinear(instance, (void**)&device.dPos[0], nbody_memsize, &allocs[0]);
        allocLinear(instance, (void**)&device.dPos[1], nbody_memsize, &allocs[1]);

        ViewDescription desc
        {
            .type   = ViewType::Markers,
            .options = {},
            .domain = DomainType::Domain3D,
            .attributes  = {
                { AttributeType::Position, {
                    .source = allocs[0],
                    .size   = input.body_count,
                    .format = FormatDescription::make<float4>(),
                }}
            },
            .layout        = Layout::make(input.body_count),
            .visible       = true,
            .default_color = {1.f, 1.f, 1.f, 1.f},
            // Under LightModel::None markers are native point sprites sized in PIXELS
            // (params.point_size is a world-space unit for the lit sphere modes and does
            // not apply). 3 px matches nbody-datoviz's MARKER_SIZE_PX for a fair visual
            // and cost comparison.
            .default_size  = 3.f,
            .linewidth     = 0.f,
            .scale         = {1.f, 1.f, 1.f},
        };
        createView(instance, &desc, &views[0]);

        desc.visible = false;
        desc.attributes[AttributeType::Position].source = allocs[1];
        createView(instance, &desc, &views[1]);
    }
    else // Run the simulation without display
    {
        checkCuda(cudaMalloc((void**)&device.dPos[0], nbody_memsize));
        checkCuda(cudaMalloc((void**)&device.dPos[1], nbody_memsize));
    }

    // Initialize simulation
    unsigned int current_read  = 0, current_write = 1;
    NBodyConfig config = NBodyConfig::Shell;
    setSofteningSquared(params.softening);
    HostData host;

    host.pos = new float[input.body_count * 4];
    host.vel = new float[input.body_count * 4];
    randomizeBodies(config, host.pos, host.vel, nullptr,
        params.cluster_scale, params.velocity_scale, input.body_count, true
    );
    checkCuda(cudaMemcpy(device.dPos[current_read], host.pos, nbody_memsize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(device.dVel, host.vel, nbody_memsize, cudaMemcpyHostToDevice));

    // Start display and measurements
    //setCameraPosition(instance, {0.f, 0.f, -3.f});
    HudData hud{ .n = (unsigned)input.body_count };
    GPUPowerBegin("gpu", 100);
    printSystemInfo(nbody_memsize, input.display);
    {
        nvmlMemory_v2_t mi;
        mi.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
        nvmlDeviceGetName(getNvmlDevice(), hud.gpu_name, sizeof(hud.gpu_name));
        hud.gpu_total_gb = (float)(mi.total / (1024.0 * 1024.0 * 1024.0));
        cudaDeviceProp prop{};
        checkCuda(cudaGetDeviceProperties(&prop, device_id));
        snprintf(hud.gpu_device, sizeof(hud.gpu_device), "%d (CC %d.%d)",
            device_id, prop.major, prop.minor);
        // VRAM breakdown: external + cuda_ctx are measured (checkpoints), the rest computed.
        hud.vram_external_mb = (float)vram_external;
        hud.cuda_ctx_mb      = (float)cuda_ctx_mb;
        hud.buf_total_mb = (float)(3 * nbody_memsize / (1024.0 * 1024.0)); // dPos[0]+dPos[1]+dVel
        hud.render_mb    = 0.f;  // point-sprite markers read the interop dPos; no extra geometry
        // Vulkan render targets mimir's swapchain allocates, no MSAA: 3 swapchain (B8G8R8A8, 4 B)
        // + 1 depth (D32_SFLOAT, 4 B), each width*height.
        const double px = (double)input.width * (double)input.height;
        hud.vulkan_mb = (float)(px * 4.0 * (3 + 1) / (1024.0 * 1024.0));
    }
    if (input.display) displayAsync(instance);

    using Clock = std::chrono::steady_clock;

    cudaEvent_t cstart = nullptr, cstop = nullptr, cstop_prev = nullptr;
    if (input.display)
    {
        checkCuda(cudaEventCreate(&cstart));
        checkCuda(cudaEventCreate(&cstop));
        if (input.enable_interop_sync)
        {
            checkCuda(cudaEventCreate(&cstop_prev));
            checkCuda(cudaEventRecord(cstop_prev, 0));
            checkCuda(cudaEventSynchronize(cstop_prev));
        }
    }

    // Main simulation loop
    // Accumulate the benchmark's own per-frame kernel time (ms). The engine's compute_monitor
    // only ticks inside prepareViews/updateViews when enable_interop_sync is on, so in async
    // mode (--interop-sync 0) getMetrics().times.compute would be 0. We feed this measured total
    // into the returned metrics below, matching samples/nbody-datoviz (compute = sum over frames,
    // in seconds) for both sync and async runs.
    double total_compute_ms = 0.0;
    // Accumulate the per-frame render-pass GPU time (getMetrics().times.pipeline is the LAST frame's
    // value -- assigned, not accumulated, by the engine). Summing it here makes pipeline_time a TOTAL in
    // seconds, consistent with compute_time and graphics_time (both totals); reading the raw last-frame
    // value gave an effectively-zero, per-frame number that did not match the other columns.
    double total_pipeline_s = 0.0;
    // Iterations actually executed (the GPU loop may stop early if the window is closed). This is
    // the number of frames the totals above accumulate over, so totals / iters_run = per-frame average.
    int iters_run = 0;
    // Wall-clock span of the whole simulate+render loop, used for a whole-run average framerate that
    // matches samples/nbody-datoviz (frames / loop_wall). The engine's getFramerate() is only a
    // trailing 240-frame window, so it would report a different number for runs where FPS drifts.
    auto loop_start = Clock::now();
    if (input.use_cpu)
    {
        host.force = new float[input.body_count * 3];
        memset(host.force, 0, input.body_count * 3 * sizeof(float));

        auto frame_start = Clock::now();
        for (int i = 0; i < input.iter_count; ++i)
        {
            iters_run = i + 1;
            // --timeout: stop the run early (still falls through to finalize + CSV below).
            if (input.timeout_s > 0.f &&
                std::chrono::duration<double>(Clock::now() - loop_start).count() >= input.timeout_s)
            { iters_run = i; break; }
            auto t0 = Clock::now();
            integrateNBodySystemCpu(host, params.time_step,
                params.damping, params.softening, input.body_count
            );
            auto t1 = Clock::now();
            total_compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (input.display) { prepareViews(instance); }
            checkCuda(cudaMemcpy(device.dPos[current_read], host.pos,
                nbody_memsize, cudaMemcpyHostToDevice)
            );
            if (input.display) { updateViews(instance); }

            if (input.display)
            {
                auto now = Clock::now();
                using ms = std::chrono::duration<float, std::milli>;
                float frame_ms = ms(now - frame_start).count();
                float new_fps  = frame_ms > 0.f ? 1000.f / frame_ms : 0.f;
                hud.frame      = i;
                hud.compute_ms = ms(t1 - t0).count();
                hud.fps        = (i == 0) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
                auto gt = getMetrics(instance).times;
                total_pipeline_s += gt.pipeline; // per-frame render-pass seconds -> accumulate to a total
                hud.wait_ms   = (i == 0) ? gt.wait   : 0.9f * hud.wait_ms   + 0.1f * gt.wait;
                hud.record_ms = (i == 0) ? gt.record : 0.9f * hud.record_ms + 0.1f * gt.record;
                hud.submit_ms = (i == 0) ? gt.submit : 0.9f * hud.submit_ms + 0.1f * gt.submit;
                hud.gpu_ms    = (i == 0) ? gt.gpu    : 0.9f * hud.gpu_ms    + 0.1f * gt.gpu;
                float watts    = (float)getGPUCurrentPower();
                hud.gpu_watts  = (i == 0) ? watts : 0.9f * hud.gpu_watts + 0.1f * watts;
                if (i == 0 || (i % 30) == 0) hud.vram_used_mb = (float)sampleVramUsedMB();
                setHudText(instance, formatHud(hud).c_str()); // push metrics to the built-in overlay
                frame_start    = now;
            }
        }

        delete[] host.force;
    }
    else
    {
        using ms = std::chrono::duration<float, std::milli>;
        auto frame_start = Clock::now();
        for (int i = 0; i < input.iter_count && isRunning(instance); ++i)
        {
            iters_run = i + 1;
            // --timeout: stop the run early (still falls through to finalize + CSV below).
            if (input.timeout_s > 0.f &&
                std::chrono::duration<double>(Clock::now() - loop_start).count() >= input.timeout_s)
            { iters_run = i; break; }
            if (input.display) { prepareViews(instance); }

            if (input.display) { checkCuda(cudaEventRecord(cstart)); }
            integrateNbodySystem(device, current_read, params.time_step,
                params.damping, input.body_count, block_size
            );
            if (input.display) { checkCuda(cudaEventRecord(cstop)); }

            std::swap(current_read, current_write);

            if (input.display)
            {
                toggleVisibility(views[0]);
                toggleVisibility(views[1]);
                updateViews(instance);

                checkCuda(cudaEventSynchronize(cstop));
                float kernel_ms = 0.f;
                checkCuda(cudaEventElapsedTime(&kernel_ms, cstart, cstop));
                total_compute_ms += kernel_ms;

                // GPU render time: cstop_prev fires at end of kernel_{i-1} (Vulkan starts),
                // cstart fires after the GPU semaphore wait in prepareViews (Vulkan done).
                // Both are on stream 0, so elapsed gives true Vulkan render latency.
                if (input.enable_interop_sync && i > 0)
                {
                    float r_ms = 0.f;
                    checkCuda(cudaEventElapsedTime(&r_ms, cstop_prev, cstart));
                    hud.render_ms = (i == 1) ? r_ms : 0.9f * hud.render_ms + 0.1f * r_ms;
                }
                if (input.enable_interop_sync) { std::swap(cstop, cstop_prev); }

                auto now       = Clock::now();
                float frame_ms = ms(now - frame_start).count();
                float new_fps  = frame_ms > 0.f ? 1000.f / frame_ms : 0.f;
                hud.frame      = i;
                hud.compute_ms = kernel_ms;
                hud.fps        = (i == 0) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
                // Render sub-costs from the engine (GPU frame latency + CPU phases), same EMA
                // smoothing as the rest of the HUD.
                auto gt = getMetrics(instance).times;
                total_pipeline_s += gt.pipeline; // per-frame render-pass seconds -> accumulate to a total
                hud.wait_ms   = (i == 0) ? gt.wait   : 0.9f * hud.wait_ms   + 0.1f * gt.wait;
                hud.record_ms = (i == 0) ? gt.record : 0.9f * hud.record_ms + 0.1f * gt.record;
                hud.submit_ms = (i == 0) ? gt.submit : 0.9f * hud.submit_ms + 0.1f * gt.submit;
                hud.gpu_ms    = (i == 0) ? gt.gpu    : 0.9f * hud.gpu_ms    + 0.1f * gt.gpu;
                float watts    = (float)getGPUCurrentPower();
                hud.gpu_watts  = (i == 0) ? watts : 0.9f * hud.gpu_watts + 0.1f * watts;
                if (i == 0 || (i % 30) == 0) hud.vram_used_mb = (float)sampleVramUsedMB();
                setHudText(instance, formatHud(hud).c_str()); // push metrics to the built-in overlay
                frame_start    = now;
            }
        }
    }

    auto loop_wall = std::chrono::duration<double>(Clock::now() - loop_start).count();

    if (cstart)     { cudaEventDestroy(cstart); }
    if (cstop)      { cudaEventDestroy(cstop); }
    if (cstop_prev) { cudaEventDestroy(cstop_prev); }

    // Retrieve metrics
    auto metrics = getMetrics(instance);
    // Override framerate with a whole-run average (frames / loop_wall), matching samples/nbody-datoviz.
    // The engine's getFramerate() averages only the last 240 frames; for a run-summary CSV we want the
    // throughput over the entire loop. iters_run == frames rendered in display mode, sim steps otherwise.
    if (loop_wall > 0.0) { metrics.frame_rate = (float)(iters_run / loop_wall); }
    // Override compute with the benchmark's own measured total (seconds). The engine only
    // populates times.compute in interop-sync mode; this makes async runs report correctly too
    // and keeps the semantics identical to samples/nbody-datoviz.
    metrics.times.compute = (float)(total_compute_ms / 1000.0);
    // Pipeline: total render-pass GPU time in seconds (see total_pipeline_s above), consistent with
    // compute/graphics. 0 in no-display mode (no render pass ran).
    metrics.times.pipeline = (float)total_pipeline_s;

    // Nvml memory report
    nvmlMemory_v2_t meminfo;
    meminfo.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
    nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);
    constexpr double gigabyte = 1024.0 * 1024.0 * 1024.0;
    GPUMemoryMetrics nvml{
        .free     = meminfo.free / gigabyte,
        .reserved = meminfo.reserved / gigabyte,
        .total    = meminfo.total / gigabyte,
        .used     = meminfo.used / gigabyte,
    };

    auto gpu_power = GPUPowerEnd();

    // Cleanup
    exit(instance);
    destroyInstance(instance);
    // dPos[0/1] with display=1 are interop pointers managed by mimir (freed via
    // allocLinear deletors inside destroyInstance); calling cudaFree on them would
    // double-free. With display=0 they are plain cudaMalloc pointers.
    if (!input.display)
    {
        checkCuda(cudaFree(device.dPos[0]));
        checkCuda(cudaFree(device.dPos[1]));
    }
    checkCuda(cudaFree(device.dVel));

    delete[] host.pos;
    delete[] host.vel;

    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml, .iters = iters_run };
}

static void usage(const char *prog)
{
    printf(
        "Usage: %s [width height] [body_count] [iters] [options]\n"
        "\n"
        "Positional (in order; width/height must be supplied together):\n"
        "  width height   Window resolution in pixels              (default: 1920 1080)\n"
        "  body_count     Number of simulated bodies               (default: 77824)\n"
        "  iters          Simulation steps to run                  (default: 1000000)\n"
        "\n"
        "Options (named, order-independent; omitted ones use their default):\n"
        "  --present N        0=Immediate 1=TripleBuffering 2=VSync (default: 0)\n"
        "                     Real display vsync lives here (--present 2).\n"
        "  --interop-sync N   CUDA-Vulkan interop sync: 1=on 0=off  (default: 1)\n"
        "                     NOT vsync; gates compute/render on the shared buffer.\n"
        "  --display N        1 = open window, 0 = simulate only    (default: 1)\n"
        "  --timeout SECS     stop after SECS wall-clock even if iters remain; the run still\n"
        "                     finalizes and writes its CSV row     (default: 0 = no timeout)\n"
        "  --use-cpu N        1 = CPU integrator, 0 = GPU kernel     (default: 0)\n"
        "\n"
        "Keys: F1 toggles the HUD (and every other GUI window) for clean screenshots;\n"
        "      Ctrl+G shows the engine scene-parameters panel; Ctrl+Q quits.\n"
        "\n"
        "Frame rate is always uncapped (no target_fps limiter).\n"
        "\n"
        "Output: one CSV row to stdout (same column layout as samples/nbody-datoviz, minus transfer_time).\n"
        "\n"
        "Examples:\n"
        "  # Open a 1920x1080 window, 1M bodies, 1000 steps, then exit:\n"
        "  %s 1920 1080 1000000 1000\n"
        "\n"
        "  # Headless simulation (no window) — measures pure compute throughput:\n"
        "  %s 1920 1080 1000000 1000 --display 0\n"
        "\n"
        "  # Use the batch driver to sweep parameters and write a CSV:\n"
        "  ./batch_main.sh results.csv\n",
        prog, prog, prog);
}

int main(int argc, char *argv[])
{
    if (argc == 1) { usage(argv[0]); return EXIT_SUCCESS; }

    auto input = BenchmarkInput::defaultValues();
    NBodyParams params = demo_params[3];

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return EXIT_SUCCESS; }
        if (a.rfind("--", 0) == 0)
        {
            if (i + 1 >= argc)
            { fprintf(stderr, "Missing value for %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
            std::string v = argv[++i];
            if      (a == "--present")      input.present = static_cast<PresentMode>(std::stoi(v));
            else if (a == "--interop-sync") input.enable_interop_sync = static_cast<bool>(std::stoi(v));
            else if (a == "--display")      input.display = static_cast<bool>(std::stoi(v));
            else if (a == "--use-cpu")      input.use_cpu = static_cast<bool>(std::stoi(v));
            else if (a == "--timeout")      input.timeout_s = std::stof(v);
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { pos.push_back(a); }
    }
    if (pos.size() >= 2) { input.width = std::stoi(pos[0]); input.height = std::stoi(pos[1]); }
    if (pos.size() >= 3)   input.body_count = std::stoul(pos[2]);
    if (pos.size() >= 4)   input.iter_count = std::stoi(pos[3]);

    auto result = runExperiment(input, params);
    formatResults(input, result);

    return EXIT_SUCCESS;
}
