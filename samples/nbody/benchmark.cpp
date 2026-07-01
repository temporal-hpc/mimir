#include <chrono>
#include <random>
#include <string> // std::stoul
#include <vector>

#include <imgui.h>

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
    else { mode = input.enable_sync? "sync" : "desync"; }

    std::string resolution = "None";
    if      (input.width == 1920 && input.height == 1080) { resolution = "FHD"; }
    else if (input.width == 2560 && input.height == 1440) { resolution = "QHD"; }
    else if (input.width == 3840 && input.height == 2160) { resolution = "UHD"; }

    auto library = result.perf;
    auto gpu = result.power;
    auto nvml = result.memory;

    // pack_time/d2h_time/h2h_time are 0 for mimir (zero-copy, no pack step).
    // Column layout matches nbody-datoviz for direct CSV comparison.
    printAligned({
        {"mode",          mode},
        {"windowres",     resolution},
        {"N",             sd(input.body_count)},
        {"target_fps",    sd(input.target_fps)},
        {"framerate",     sf(library.frame_rate)},
        {"compute_time",  sf(library.times.compute)},
        {"pipeline_time", sf(library.times.pipeline)},
        {"graphics_time", sf(library.times.graphics)},
        {"vk_usage",      sf(library.devmem.usage)},
        {"vk_budget",     sf(library.devmem.budget)},
        {"gpu_power",     sf(gpu.average_power)},
        {"gpu_energy",    sf(gpu.total_energy)},
        {"gpu_time",      sf(gpu.total_time)},
        {"nvml_free",     sf(nvml.free)},
        {"nvml_reserved", sf(nvml.reserved)},
        {"nvml_total",    sf(nvml.total)},
        {"nvml_used",     sf(nvml.used)},
        {"pack_time",     sf(0.f)},
        {"d2h_time",      sf(0.f)},
        {"h2h_time",      sf(0.f)},
    });
    printf("%s,%s,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(),
        resolution.c_str(),
        input.body_count,
        input.target_fps,
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
    char         gpu_name[256] = {};
    float        gpu_total_gb  = 0.f;
    float        buf_total_mb  = 0.f;
};

BenchmarkResult runExperiment(BenchmarkInput input, NBodyParams params)
{
    params.point_size /= 5.f;

    // CUDA initialization
    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));
    // Kernel block size
    constexpr unsigned int block_size = 256;

    if (input.display == false) { input.width = input.height = 1; }

    ViewerOptions options{};
    options.window.size = {input.width, input.height}; // Starting window size
    options.background_color = {0.f, 0.f, 0.f, 1.f};
    options.present.mode = input.present;
    options.present.enable_sync = input.enable_sync;
    options.present.target_fps  = input.target_fps;
    options.show_panel = input.display; // required for setGuiCallback to fire

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
            .default_size  = params.point_size / 5.f,
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

    // Decrease the time step further based on the number of bodies
    // This is to prevent making them move outside the camera area
    if (input.body_count <= 10000)
    {
        params.time_step /= 1000.f;
        setCameraPosition(instance, {0.f, 0.f, -1.3f});
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
    if (input.display)
    {
        setGuiCallback(instance, [&hud]() {
            ImVec2 disp = ImGui::GetIO().DisplaySize;
            // Pivot (1,0) anchors the window's top-right corner to this position,
            // so the window auto-sizes leftward without clipping at the right edge.
            ImGui::SetNextWindowPos(ImVec2(disp.x - 10.f, 10.f), ImGuiCond_Always, ImVec2(1.f, 0.f));
            ImGui::Begin("Performance", nullptr,
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_AlwaysAutoResize);
            if (ImGui::BeginTable("hw", 2)) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("GPU");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%s", hud.gpu_name);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("VRAM");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f GB", hud.gpu_total_gb);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Buffers");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f MB", hud.buf_total_mb);
                ImGui::EndTable();
            }
            ImGui::Separator();
            if (ImGui::BeginTable("perf", 2)) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Bodies");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", hud.n);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Frame");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d", hud.frame);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("FPS");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f", hud.fps);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Compute");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.compute_ms);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Pack");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("D2H");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("H2H");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Transfer");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A (pack + D2H + H2H)");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Render");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.render_ms);
                ImGui::EndTable();
            }
            ImGui::End();
        });
    }
    GPUPowerBegin("gpu", 100);
    printSystemInfo(nbody_memsize, input.display);
    {
        nvmlMemory_v2_t mi;
        mi.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
        nvmlDeviceGetName(getNvmlDevice(), hud.gpu_name, sizeof(hud.gpu_name));
        hud.gpu_total_gb = (float)(mi.total / (1024.0 * 1024.0 * 1024.0));
        hud.buf_total_mb = (float)(3 * nbody_memsize / (1024.0 * 1024.0));
    }
    if (input.display) displayAsync(instance);

    using Clock = std::chrono::steady_clock;

    cudaEvent_t cstart = nullptr, cstop = nullptr, cstop_prev = nullptr;
    if (input.display)
    {
        checkCuda(cudaEventCreate(&cstart));
        checkCuda(cudaEventCreate(&cstop));
        if (input.enable_sync)
        {
            checkCuda(cudaEventCreate(&cstop_prev));
            checkCuda(cudaEventRecord(cstop_prev, 0));
            checkCuda(cudaEventSynchronize(cstop_prev));
        }
    }

    // Main simulation loop
    if (input.use_cpu)
    {
        host.force = new float[input.body_count * 3];
        memset(host.force, 0, input.body_count * 3 * sizeof(float));

        auto frame_start = Clock::now();
        for (int i = 0; i < input.iter_count; ++i)
        {
            auto t0 = Clock::now();
            integrateNBodySystemCpu(host, params.time_step,
                params.damping, params.softening, input.body_count
            );
            auto t1 = Clock::now();
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

                // GPU render time: cstop_prev fires at end of kernel_{i-1} (Vulkan starts),
                // cstart fires after the GPU semaphore wait in prepareViews (Vulkan done).
                // Both are on stream 0, so elapsed gives true Vulkan render latency.
                if (input.enable_sync && i > 0)
                {
                    float r_ms = 0.f;
                    checkCuda(cudaEventElapsedTime(&r_ms, cstop_prev, cstart));
                    hud.render_ms = (i == 1) ? r_ms : 0.9f * hud.render_ms + 0.1f * r_ms;
                }
                if (input.enable_sync) { std::swap(cstop, cstop_prev); }

                auto now       = Clock::now();
                float frame_ms = ms(now - frame_start).count();
                float new_fps  = frame_ms > 0.f ? 1000.f / frame_ms : 0.f;
                hud.frame      = i;
                hud.compute_ms = kernel_ms;
                hud.fps        = (i == 0) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
                frame_start    = now;
            }
        }
    }

    if (cstart)     { cudaEventDestroy(cstart); }
    if (cstop)      { cudaEventDestroy(cstop); }
    if (cstop_prev) { cudaEventDestroy(cstop_prev); }

    // Retrieve metrics
    auto metrics = getMetrics(instance);

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

    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml };
}

static void usage(const char *prog)
{
    printf(
        "Usage: %s [width height] [body_count] [iters] [present] [target_fps] [vsync] [display] [use_cpu]\n"
        "\n"
        "  width height   Window resolution in pixels              (default: 1920 1080)\n"
        "  body_count     Number of simulated bodies               (default: 77824)\n"
        "  iters          Simulation steps to run                  (default: 1000000)\n"
        "  present        0=Immediate, 1=TripleBuffering, 2=VSync  (default: 0)\n"
        "  target_fps     Frame-rate cap; 0 = unlimited            (default: 0)\n"
        "  vsync          1 = enable VSync, 0 = disable            (default: 1)\n"
        "  display        1 = open window and render, 0 = simulate only (no window) (default: 1)\n"
        "  use_cpu        1 = CPU integrator, 0 = GPU kernel       (default: 0)\n"
        "\n"
        "All arguments are positional and optional; omitted trailing args use their defaults.\n"
        "width and height must be supplied together.\n"
        "\n"
        "Output: one CSV row to stdout (same column layout as samples/nbody-datoviz, minus transfer_time).\n"
        "\n"
        "Examples:\n"
        "  # Open a 1920x1080 window, 1M bodies, 1000 steps, then exit:\n"
        "  %s 1920 1080 1000000 1000\n"
        "\n"
        "  # Headless simulation (no window) — measures pure compute throughput:\n"
        "  %s 1920 1080 1000000 1000 0 0 1 0\n"
        "\n"
        "  # Use the batch driver to sweep parameters and write a CSV:\n"
        "  ./batch_main.sh results.csv\n",
        prog, prog, prog);
}

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    if (argc == 1) { usage(argv[0]); return EXIT_SUCCESS; }

    auto input = BenchmarkInput::defaultValues();
    NBodyParams params = demo_params[3];

    // Parse parameters from command line
    if (argc >= 3) { input.width = std::stoi(argv[1]); input.height = std::stoi(argv[2]); }
    if (argc >= 4) input.body_count  = std::stoul(argv[3]);
    if (argc >= 5) input.iter_count  = std::stoi(argv[4]);
    if (argc >= 6) input.present     = static_cast<PresentMode>(std::stoi(argv[5]));
    if (argc >= 7) input.target_fps  = std::stoi(argv[6]);
    if (argc >= 8) input.enable_sync = static_cast<bool>(std::stoi(argv[7]));
    if (argc >= 9) input.display     = static_cast<bool>(std::stoi(argv[8]));
    if (argc >= 10) input.use_cpu    = static_cast<bool>(std::stoi(argv[9]));

    auto result = runExperiment(input, params);
    formatResults(input, result);

    return EXIT_SUCCESS;
}
