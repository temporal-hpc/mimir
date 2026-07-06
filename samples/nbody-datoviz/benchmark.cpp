// N-body benchmark rendered with datoviz instead of mimir.
//
// This is a deliberate, metric-for-metric port of samples/nbody/benchmark.cpp so the two
// libraries can be compared head-to-head. The physics (kernel, CPU integrator, body
// initialization, parameters, RNG seed) are identical. The only difference is the
// rendering path:
//
//   mimir   : CUDA writes positions into a Vulkan/CUDA *shared* buffer (allocLinear).
//             Zero per-frame transfer; the renderer reads the simulation result in place.
//   datoviz : has no CUDA interop (roadmap v0.4+). Its API only accepts host pointers, so
//             each frame the positions must round-trip GPU -> Host -> GPU:
//               1. pack (GPU kernel)  float4 -> vec3 (smaller transfer, no host repack),
//               2. D2H  (PCIe DMA)    device -> pinned host,
//               3. H2H  (CPU memcpy)  pinned host -> datoviz's internal heap copy,
//                                     inside dvz_marker_position,
//               4. H2D + D2D + draw  staging write + GPU upload + draw,
//                                     inside dvz_scene_step (inseparable).
//             That round trip is the overhead this benchmark quantifies (transfer_time).

#include <atomic>
#include <chrono>
#include <cstdlib> // setenv
#include <cstring> // memset
#include <random>
#include <string> // std::stoul
#include <vector> // std::vector

#include <datoviz.h>
#include <imgui.h>

#include "benchmark.hpp"
#include "nbody_gpu.cuh"
#include "nbody_cpu.hpp"
#include "pack.cuh"
#include "nvmlPower.hpp"

using clk = std::chrono::high_resolution_clock;
static inline double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct NBodyParams
{
    float time_step;
    float cluster_scale;
    float velocity_scale;
    float softening;
    float damping;
    float point_size;
    float x, y, z;
};

// Same demo parameter table as samples/nbody (we use demo_params[3], matching nbody).
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

// Performance metrics, mirroring mimir's PerformanceMetrics so the CSV columns line up.
// pipeline (GPU render-pass time) is not measurable through datoviz's public API and is
// reported as 0. devmem usage/budget are substituted with NVML used/total (see formatResults).
// graphics_time = dvz_scene_step wall-clock (H2D staging write + D2D upload + draw).
// transfer.{pack,d2h,h2h} are the three measurable stages before dvz_scene_step.
struct PerformanceMetrics {
    float frame_rate;
    struct { float compute; float graphics; float pipeline; } times;
    struct { float usage; float budget; } devmem;
    struct { float pack; float d2h; float h2h; } transfer;
};

static std::string sf(float v)  { char b[32]; snprintf(b, sizeof(b), "%f", v); return b; }
static std::string sd(int v)    { return std::to_string(v); }
static std::string smb(size_t bytes) {
    char b[32]; snprintf(b, sizeof(b), "%.2f MB", bytes / (1024.0 * 1024.0)); return b;
}

static void printSystemInfo(size_t nbody_memsize, size_t vec3_memsize)
{
    char gpu_name[256] = "Unknown";
    nvmlDeviceGetName(getNvmlDevice(), gpu_name, sizeof(gpu_name));

    nvmlMemory_v2_t mi;
    mi.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
    nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
    constexpr double gb = 1024.0 * 1024.0 * 1024.0;

    fprintf(stderr, "GPU: %s\n", gpu_name);
    fprintf(stderr, "Total GPU memory: %.2f GB\n", mi.total / gb);
    fprintf(stderr, "Buffers:\n");
    fprintf(stderr, "  dPos[0]  (CUDA):   %s\n", smb(nbody_memsize).c_str());
    fprintf(stderr, "  dPos[1]  (CUDA):   %s\n", smb(nbody_memsize).c_str());
    fprintf(stderr, "  dVel     (CUDA):   %s\n", smb(nbody_memsize).c_str());
    fprintf(stderr, "  d_pos3   (CUDA):   %s\n", smb(vec3_memsize).c_str());
    fprintf(stderr, "  h_pos3   (pinned): %s\n", smb(vec3_memsize).c_str());
    fprintf(stderr, "  Total:             %s\n", smb(3 * nbody_memsize + 2 * vec3_memsize).c_str());
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

struct BenchmarkResult {
    PerformanceMetrics perf;
    GPUPowerMetrics power;
    GPUMemoryMetrics memory;
};

void formatResults(BenchmarkInput input, BenchmarkResult result)
{
    // Mode label parallels nbody: "datoviz" when rendering, "none" for pure simulation.
    std::string mode = input.display ? "datoviz" : "none";

    std::string resolution = "None";
    if      (input.width == 1920 && input.height == 1080) { resolution = "FHD"; }
    else if (input.width == 2560 && input.height == 1440) { resolution = "QHD"; }
    else if (input.width == 3840 && input.height == 2160) { resolution = "UHD"; }

    auto library = result.perf;
    auto gpu = result.power;
    auto nvml = result.memory;

    // Same column order as nbody.
    // graphics_time = dvz_scene_step (H2D staging write + D2D upload + draw, inseparable).
    // pack_time/d2h_time/h2h_time are the three measurable transfer stages.
    printAligned({
        {"mode",          mode},
        {"windowres",     resolution},
        {"N",             sd(input.body_count)},
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
        {"pack_time",     sf(library.transfer.pack)},
        {"d2h_time",      sf(library.transfer.d2h)},
        {"h2h_time",      sf(library.transfer.h2h)},
    });
    printf("%s,%s,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(),
        resolution.c_str(),
        input.body_count,
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
        library.transfer.pack,
        library.transfer.d2h,
        library.transfer.h2h
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

// Identical body initialization to samples/nbody (same RNG seed 12345 -> same scene).
void randomizeBodies(NBodyConfig config, float *pos, float *vel, float *color,
    float cluster_scale, float velocity_scale, int body_count, bool vec4vel)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rand_pos(-1.f, 1.f);
    std::uniform_real_distribution<float> rand_unit(0.f, 1.f);

    float mass = 1.f;
    float inv_mass = 1.f;
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

                pos[p++] = point.x * scale;
                pos[p++] = point.y * scale;
                pos[p++] = point.z * scale;
                pos[p++] = mass;

                vel[v++] = velocity.x * vscale;
                vel[v++] = velocity.y * vscale;
                vel[v++] = velocity.z * vscale;
                if (vec4vel) vel[v++] = inv_mass;

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

                x = 0.0f;
                y = 0.0f;
                z = 1.0f;
                float3 axis = {x, y, z};
                normalize(axis);

                if (1 - dot(point, axis) < 1e-6)
                {
                    axis.x = point.y;
                    axis.y = point.x;
                    normalize(axis);
                }

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

                pos[p++] = point.x * scale;
                pos[p++] = point.y * scale;
                pos[p++] = point.z * scale;
                pos[p++] = mass;
                vel[v++] = point.x * vscale;
                vel[v++] = point.y * vscale;
                vel[v++] = point.z * vscale;
                if (vec4vel) vel[v++] = inv_mass;

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

// Live per-frame metrics written by the simulation loop, read by the GUI callback.
struct HudData
{
    unsigned int n;
    int          frame;
    float        fps;
    float        compute_ms;
    float        pack_ms;      // packPositionsVec3 GPU kernel (or CPU repack)
    float        d2h_ms;       // cudaMemcpy Device→Host (PCIe DMA; 0 in CPU path)
    float        h2h_ms;       // dvz_marker_position (pinned host → datoviz internal heap copy)
    float        render_ms;    // dvz_scene_step: H2D staging write + D2D upload + draw (inseparable)
    float        gpu_watts;
    char         gpu_name[256];
    char         gpu_device[64];   // "N (CC major.minor)"
    float        gpu_total_gb;  // total VRAM (NVML)
    // VRAM used (NVML) dismembered so the sub-parts sum to it. External/CUDA are anchored on
    // measured NVML checkpoints at startup; Render/Vulkan are computed; Datoviz is the remainder.
    float        vram_used_mb;
    float        vram_external_mb;  // measured: NVML used before we touch the GPU (other processes)
    float        cuda_ctx_mb;       // measured: CUDA context reservation (cudaFree(0) checkpoint)
    float        buf_total_mb;      // computed: our CUDA sim device buffers (dPos x2 + dVel + d_pos3)
    float        render_mb;         // computed: datoviz marker vertices (pos + size + color)
    float        vulkan_mb;         // computed: Vulkan render targets (swapchain + depth + staging)
};

// HUD visibility, toggled with F1 (clean-viewport screenshots). Written by the keyboard
// callback, read by the GUI callback.
static std::atomic<bool> g_show_hud{true};

// Keyboard callback: F1 toggles the HUD.
static void keyCallback(DvzApp* /*app*/, DvzId /*window_id*/, DvzKeyboardEvent* ev)
{
    if (ev->type == DVZ_KEYBOARD_EVENT_PRESS && ev->key == DVZ_KEY_F1)
    {
        g_show_hud.store(!g_show_hud.load());
    }
}

static void hudCallback(DvzApp* /*app*/, DvzId /*canvas_id*/, DvzGuiEvent* ev)
{
    if (!g_show_hud.load()) { return; }
    auto* hud = static_cast<HudData*>(ev->user_data);
    // Borderless overlay in the top-right corner, matching the mimir benchmarks' HUD.
    dvz_gui_corner(DVZ_DIALOG_CORNER_TOP_RIGHT, (vec2){10, 10});
    dvz_gui_begin("Datoviz - nbody", DVZ_DIALOG_FLAGS_OVERLAY);
    dvz_gui_text("GPU        %s",       hud->gpu_name);
    dvz_gui_text("Device     %s",       hud->gpu_device);
    dvz_gui_text("VRAM       %.1f GB",  hud->gpu_total_gb);
    // VRAM used (NVML, whole GPU) fully dismembered; the six sub-lines sum to it by construction.
    // External + CUDA ctx are anchored on measured NVML checkpoints; CUDA buf + Render + Vulkan are
    // computed; Datoviz is the remainder = datoviz's own device structures (buffer pools/pipelines).
    float datoviz_mb = hud->vram_used_mb - hud->vram_external_mb - hud->cuda_ctx_mb
                     - hud->buf_total_mb - hud->render_mb - hud->vulkan_mb;
    dvz_gui_text("VRAM used         %.0f MB", hud->vram_used_mb);
    dvz_gui_text("  External procs  %.0f MB", hud->vram_external_mb); // other processes (measured)
    dvz_gui_text("  CUDA context    %.0f MB", hud->cuda_ctx_mb);      // CUDA runtime reserve (measured)
    dvz_gui_text("  CUDA buffers    %.1f MB", hud->buf_total_mb);     // our sim device buffers (computed)
    dvz_gui_text("  Render geometry %.1f MB", hud->render_mb);        // datoviz marker vertices (computed)
    dvz_gui_text("  Vulkan targets  %.0f MB", hud->vulkan_mb);        // swapchain + depth + staging
    dvz_gui_text("  Datoviz structs %.0f MB", datoviz_mb);           // datoviz buffer pools/pipelines
    dvz_gui_text("");
    dvz_gui_text("Bodies     %u",       hud->n);
    dvz_gui_text("Frame      %d",       hud->frame);
    dvz_gui_text("FPS        %.1f",     hud->fps);
    dvz_gui_text("Compute    %.2f ms",  hud->compute_ms);
    dvz_gui_text("Transfer   %.2f ms",  hud->pack_ms + hud->d2h_ms + hud->h2h_ms);
    dvz_gui_text("    Pack   %.2f ms",  hud->pack_ms);
    dvz_gui_text("    D2H    %.2f ms",  hud->d2h_ms);
    dvz_gui_text("    H2H    %.2f ms",  hud->h2h_ms);
    dvz_gui_text("Render     %.2f ms (H2D + D2D + draw)", hud->render_ms);
    dvz_gui_text("Power      %.1f W",   hud->gpu_watts);
    dvz_gui_end();
}

// Holds the datoviz objects so setup/teardown stay tidy.
struct DatovizContext
{
    DvzApp *app;
    DvzBatch *batch;
    DvzScene *scene;
    DvzFigure *figure;
    DvzPanel *panel;
    DvzArcball *arcball;
    DvzVisual *visual;
};

// Marker size in pixels. datoviz sizes are in screen pixels; the NBodyParams::point_size
// field is in world-space units tuned for mimir and is not reusable here.
static constexpr float MARKER_SIZE_PX = 3.0f;

static DatovizContext setupDatoviz(BenchmarkInput input, const float *initial_pos3, unsigned int n)
{
    DatovizContext ctx{};

    ctx.app = dvz_app(DVZ_APP_FLAGS_NONE);
    ctx.batch = dvz_app_batch(ctx.app);
    ctx.scene = dvz_scene(ctx.batch);

    int fig_flags = DVZ_CANVAS_FLAGS_IMGUI;
    if (input.vsync) fig_flags |= DVZ_CANVAS_FLAGS_VSYNC;
    ctx.figure = dvz_figure(ctx.scene, input.width, input.height, fig_flags);
    ctx.panel = dvz_panel_default(ctx.figure);
    ctx.arcball = dvz_panel_arcball(ctx.panel, 0); // 3D interactivity

    // Filled disc markers, matching mimir's Markers view appearance.
    ctx.visual = dvz_marker(ctx.batch, 0);
    dvz_marker_mode(ctx.visual, DVZ_MARKER_MODE_CODE);
    dvz_marker_aspect(ctx.visual, DVZ_MARKER_ASPECT_FILLED);
    dvz_marker_shape(ctx.visual, DVZ_MARKER_SHAPE_DISC);
    dvz_marker_alloc(ctx.visual, n);

    dvz_marker_position(ctx.visual, 0, n, (vec3 *)initial_pos3, 0);

    std::vector<float> sizes(n, MARKER_SIZE_PX);
    dvz_marker_size(ctx.visual, 0, n, sizes.data(), 0);

    std::vector<DvzColor> colors(n);
    for (unsigned int i = 0; i < n; i++)
    {
        colors[i][0] = 255; colors[i][1] = 255; colors[i][2] = 255; colors[i][3] = 255;
    }
    dvz_marker_color(ctx.visual, 0, n, colors.data(), 0);

    dvz_panel_visual(ctx.panel, ctx.visual, 0);
    return ctx;
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

    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));
    // VRAM checkpoint #2: force the primary CUDA context to exist (cudaFree(0)) and measure its
    // reservation on its own, before our buffers or the Vulkan renderer are created.
    checkCuda(cudaFree(0));
    double cuda_ctx_mb = sampleVramUsedMB() - vram_external;
    if (cuda_ctx_mb < 0.0) cuda_ctx_mb = 0.0;
    constexpr unsigned int block_size = 256;

    const unsigned int n = input.body_count;
    auto nbody_memsize = sizeof(float4) * n;
    auto vec3_memsize  = sizeof(float3) * n;

    // Plain (non-interop) device buffers: this is the whole point of the comparison.
    DeviceData device;
    checkCuda(cudaMalloc((void **)&device.dPos[0], nbody_memsize));
    checkCuda(cudaMalloc((void **)&device.dPos[1], nbody_memsize));
    checkCuda(cudaMalloc((void **)&device.dVel, nbody_memsize));

    // Packed vec3 device buffer + pinned host staging buffer for fast D2H.
    float *d_pos3 = nullptr;
    float *h_pos3 = nullptr;
    checkCuda(cudaMalloc((void **)&d_pos3, vec3_memsize));
    checkCuda(cudaHostAlloc((void **)&h_pos3, vec3_memsize, cudaHostAllocDefault));

    // Initialize bodies (identical to nbody).
    unsigned int current_read = 0, current_write = 1;
    NBodyConfig config = NBodyConfig::Shell;
    setSofteningSquared(params.softening);

    HostData host;
    host.pos = new float[n * 4];
    host.vel = new float[n * 4];
    randomizeBodies(config, host.pos, host.vel, nullptr,
        params.cluster_scale, params.velocity_scale, n, true
    );
    checkCuda(cudaMemcpy(device.dPos[current_read], host.pos, nbody_memsize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(device.dVel, host.vel, nbody_memsize, cudaMemcpyHostToDevice));

    // Prime the packed host buffer with the initial positions for datoviz setup.
    packPositionsVec3(device.dPos[current_read], d_pos3, n, block_size);
    checkCuda(cudaMemcpy(h_pos3, d_pos3, vec3_memsize, cudaMemcpyDeviceToHost));

    DatovizContext ctx{};
    HudData hud{};
    hud.n = n;
    if (input.display)
    {
        ctx = setupDatoviz(input, h_pos3, n);
        dvz_app_gui(ctx.app, dvz_figure_id(ctx.figure), hudCallback, &hud);
        dvz_app_on_keyboard(ctx.app, keyCallback, nullptr); // F1: toggle the HUD
    }

    // CUDA events: cstart/cstop for physics kernel, pstart/pstop for pack kernel.
    cudaEvent_t cstart, cstop, pstart, pstop;
    checkCuda(cudaEventCreate(&cstart));
    checkCuda(cudaEventCreate(&cstop));
    checkCuda(cudaEventCreate(&pstart));
    checkCuda(cudaEventCreate(&pstop));

    float total_compute = 0.f, total_pack = 0.f, total_d2h = 0.f,
          total_h2h = 0.f, total_graphics = 0.f;
    size_t frame_count = 0;

    GPUPowerBegin("gpu", 100);
    printSystemInfo(nbody_memsize, vec3_memsize);
    {
        nvmlMemory_v2_t mi;
        mi.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
        nvmlDeviceGetName(getNvmlDevice(), hud.gpu_name, sizeof(hud.gpu_name));
        hud.gpu_total_gb = (float)(mi.total / (1024.0 * 1024.0 * 1024.0));
        int device_id = -1;
        checkCuda(cudaGetDevice(&device_id));
        cudaDeviceProp prop{};
        checkCuda(cudaGetDeviceProperties(&prop, device_id));
        snprintf(hud.gpu_device, sizeof(hud.gpu_device), "%d (CC %d.%d)",
            device_id, prop.major, prop.minor);
        // VRAM breakdown: external + cuda_ctx are measured (checkpoints), the rest computed.
        hud.vram_external_mb = (float)vram_external;
        hud.cuda_ctx_mb      = (float)cuda_ctx_mb;
        // Device sim buffers (VRAM only): dPos[0]+dPos[1]+dVel (float4) + d_pos3 (float3). The
        // pinned host h_pos3 is host RAM, not VRAM, so it is excluded.
        hud.buf_total_mb = (float)((3 * nbody_memsize + vec3_memsize) / (1024.0 * 1024.0));
        // datoviz marker vertices: 1/particle, position(12) + size(4) + color(4).
        hud.render_mb = (float)((double)n * (sizeof(float3) + sizeof(float) + 4) / (1024.0 * 1024.0));
        // Vulkan render targets datoviz's canvas allocates, no MSAA: 3 swapchain + 1 staging color
        // (B8G8R8A8, 4 B) + 3 depth (D32_SFLOAT, 4 B), each width*height.
        const double px = (double)input.width * (double)input.height;
        hud.vulkan_mb = (float)(px * 4.0 * (4 + 3) / (1024.0 * 1024.0));
    }
    auto loop_start = clk::now();

    if (input.use_cpu)
    {
        host.force = new float[n * 3];
        memset(host.force, 0, n * 3 * sizeof(float));

        for (int i = 0; i < input.iter_count; ++i)
        {
            auto c0 = clk::now();
            integrateNBodySystemCpu(host, params.time_step,
                params.damping, params.softening, n
            );
            total_compute += (float)ms_since(c0);

            if (input.display)
            {
                // Pack: CPU repack float4→vec3 (no GPU kernel in CPU path, d2h=0).
                auto t_pack = clk::now();
                for (unsigned int b = 0; b < n; b++)
                {
                    h_pos3[3 * b + 0] = host.pos[4 * b + 0];
                    h_pos3[3 * b + 1] = host.pos[4 * b + 1];
                    h_pos3[3 * b + 2] = host.pos[4 * b + 2];
                }
                float pack_ms = (float)ms_since(t_pack);

                // H2H: pinned host → datoviz's internal heap copy (CPU memcpy).
                auto t_h2h = clk::now();
                dvz_marker_position(ctx.visual, 0, n, (vec3 *)h_pos3, 0);
                float h2h_ms = (float)ms_since(t_h2h);

                total_pack += pack_ms;
                total_h2h  += h2h_ms;

                // Render: dvz_scene_step = H2D staging write + D2D upload + draw.
                auto t_render = clk::now();
                if (!dvz_scene_step(ctx.scene, ctx.app)) { break; }
                float graphics_ms = (float)ms_since(t_render);
                total_graphics += graphics_ms;
                frame_count++;

                float frame_compute = (float)ms_since(c0) - pack_ms - h2h_ms - graphics_ms;
                float frame_total   = frame_compute + pack_ms + h2h_ms + graphics_ms;
                hud.frame      = (int)frame_count;
                hud.compute_ms = frame_compute;
                hud.pack_ms    = pack_ms;
                hud.d2h_ms     = 0.f;
                hud.h2h_ms     = h2h_ms;
                hud.render_ms  = graphics_ms;
                if (frame_total > 0) {
                    float new_fps = 1000.f / frame_total;
                    hud.fps = (frame_count == 1) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
                }
                float watts   = (float)getGPUCurrentPower();
                hud.gpu_watts = (frame_count == 1) ? watts : 0.9f * hud.gpu_watts + 0.1f * watts;
                if (frame_count == 1 || (frame_count % 30) == 0)
                    hud.vram_used_mb = (float)sampleVramUsedMB();
            }
        }
        delete[] host.force;
    }
    else
    {
        for (int i = 0; i < input.iter_count; ++i)
        {
            // --- Compute (physics kernel only, same as mimir's compute metric) ---
            checkCuda(cudaEventRecord(cstart));
            integrateNbodySystem(device, current_read, params.time_step,
                params.damping, n, block_size
            );
            checkCuda(cudaEventRecord(cstop));
            checkCuda(cudaEventSynchronize(cstop));
            float compute_ms = 0.f;
            checkCuda(cudaEventElapsedTime(&compute_ms, cstart, cstop));
            total_compute += compute_ms;

            std::swap(current_read, current_write);

            if (input.display)
            {
                // Pack: float4 → vec3 device buffer (GPU kernel).
                checkCuda(cudaEventRecord(pstart));
                packPositionsVec3(device.dPos[current_read], d_pos3, n, block_size);
                checkCuda(cudaEventRecord(pstop));

                // D2H: device → pinned host (synchronous; also drains stream 0,
                //      making pstop queryable immediately after).
                auto t_d2h = clk::now();
                checkCuda(cudaMemcpy(h_pos3, d_pos3, vec3_memsize, cudaMemcpyDeviceToHost));
                float d2h_ms = (float)ms_since(t_d2h);

                float pack_ms = 0.f;
                checkCuda(cudaEventElapsedTime(&pack_ms, pstart, pstop));

                // H2H: pinned host → datoviz's internal heap copy (CPU memcpy).
                auto t_h2h = clk::now();
                dvz_marker_position(ctx.visual, 0, n, (vec3 *)h_pos3, 0);
                float h2h_ms = (float)ms_since(t_h2h);

                total_pack += pack_ms;
                total_d2h  += d2h_ms;
                total_h2h  += h2h_ms;

                // Render: dvz_scene_step = H2D staging write + D2D upload + draw.
                // These operations are inseparable via the datoviz public API.
                auto t_render = clk::now();
                if (!dvz_scene_step(ctx.scene, ctx.app)) { break; }
                float graphics_ms = (float)ms_since(t_render);
                total_graphics += graphics_ms;
                frame_count++;

                hud.frame      = (int)frame_count;
                hud.compute_ms = compute_ms;
                hud.pack_ms    = pack_ms;
                hud.d2h_ms     = d2h_ms;
                hud.h2h_ms     = h2h_ms;
                hud.render_ms  = graphics_ms;
                float frame_total = compute_ms + pack_ms + d2h_ms + h2h_ms + graphics_ms;
                if (frame_total > 0) {
                    float new_fps = 1000.f / frame_total;
                    hud.fps = (frame_count == 1) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
                }
                float watts   = (float)getGPUCurrentPower();
                hud.gpu_watts = (frame_count == 1) ? watts : 0.9f * hud.gpu_watts + 0.1f * watts;
                if (frame_count == 1 || (frame_count % 30) == 0)
                    hud.vram_used_mb = (float)sampleVramUsedMB();
            }
        }
    }

    auto loop_wall = std::chrono::duration<double>(clk::now() - loop_start).count();

    // Framerate: realtime throughput of the whole simulate+render loop.
    float frame_rate = 0.f;
    if (input.display && loop_wall > 0.0) { frame_rate = (float)(frame_count / loop_wall); }
    else if (loop_wall > 0.0)            { frame_rate = (float)(input.iter_count / loop_wall); }

    // NVML memory report (identical to nbody).
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

    PerformanceMetrics metrics{};
    metrics.frame_rate           = frame_rate;
    metrics.times.compute        = total_compute  / 1000.f; // ms → s
    metrics.times.graphics       = total_graphics / 1000.f;
    metrics.times.pipeline       = 0.f;
    metrics.devmem.usage         = (float)nvml.used;
    metrics.devmem.budget        = (float)nvml.total;
    metrics.transfer.pack        = total_pack / 1000.f;
    metrics.transfer.d2h         = total_d2h  / 1000.f;
    metrics.transfer.h2h         = total_h2h  / 1000.f;

    // Cleanup
    checkCuda(cudaEventDestroy(cstart));
    checkCuda(cudaEventDestroy(cstop));
    checkCuda(cudaEventDestroy(pstart));
    checkCuda(cudaEventDestroy(pstop));
    if (input.display)
    {
        dvz_scene_destroy(ctx.scene);
        dvz_app_destroy(ctx.app);
    }
    checkCuda(cudaFree(device.dPos[0]));
    checkCuda(cudaFree(device.dPos[1]));
    checkCuda(cudaFree(device.dVel));
    checkCuda(cudaFree(d_pos3));
    checkCuda(cudaFreeHost(h_pos3));
    delete[] host.pos;
    delete[] host.vel;

    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml };
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
        "  --vsync N          display vsync: 1=on 0=off             (default: 1)\n"
        "  --display N        1 = open window, 0 = simulate only    (default: 1)\n"
        "  --use-cpu N        1 = CPU integrator, 0 = GPU kernel     (default: 0)\n"
        "\n"
        "(datoviz has no present-mode selection; the mimir --present flag has no\n"
        " datoviz equivalent, so it is intentionally absent here.)\n"
        "Keys: F1 toggles the HUD for clean screenshots.\n"
        "Frame rate is always uncapped (no target_fps limiter).\n"
        "\n"
        "Output: one CSV row to stdout (same column layout as samples/nbody, plus transfer_time).\n"
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
            if      (a == "--vsync")   input.vsync   = static_cast<bool>(std::stoi(v));
            else if (a == "--display") input.display = static_cast<bool>(std::stoi(v));
            else if (a == "--use-cpu") input.use_cpu = static_cast<bool>(std::stoi(v));
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { pos.push_back(a); }
    }
    if (pos.size() >= 2) { input.width = std::stoi(pos[0]); input.height = std::stoi(pos[1]); }
    if (pos.size() >= 3)   input.body_count = std::stoul(pos[2]);
    if (pos.size() >= 4)   input.iter_count = std::stoi(pos[3]);

    // CUDA runs on device 0; render on the same GPU instead of datoviz's own "best GPU"
    // pick, which can land on a different card in multi-GPU systems (the benchmark would
    // then measure cross-GPU traffic, and the pick may lack a swapchain). Vulkan and CUDA
    // enumeration order can still differ; set DVZ_GPU=<idx> explicitly if they do.
    setenv("DVZ_GPU", "0", /*overwrite=*/0);

    auto result = runExperiment(input, params);
    formatResults(input, result);

    return EXIT_SUCCESS;
}
