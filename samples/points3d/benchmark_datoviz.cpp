// 3D random-walk point cloud benchmark — datoviz rendering path.
//
// Identical simulation to benchmark_mimir.cpp (same kernels, same seed, same
// [-1,1]^3 domain). The only difference is the rendering path:
//
//   mimir   : CUDA integrates positions in a Vulkan/CUDA shared buffer — zero transfer.
//   datoviz : has no CUDA interop; each frame the position buffer must round-trip:
//               1. D2H    float3 positions (device) -> pinned host buffer  (PCIe DMA)
//               2. H2H    pinned host -> datoviz's internal heap copy      (CPU memcpy,
//                                                                           inside dvz_marker_position)
//               3. H2D + D2D + draw   staging write + GPU upload + draw    (inside
//                                                                           dvz_scene_step, inseparable)
//
// The simulation already produces tightly packed float3 positions, so no pack
// kernel is needed (pack_time is always 0 — same as benchmark_mimir).

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include <datoviz.h>

#include "points3d_sim.cuh"
#include "nvmlPower.hpp"
#include "validation.hpp"

using clk = std::chrono::high_resolution_clock;
static inline double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Input / output structs
// ---------------------------------------------------------------------------

struct PointsInput {
    int          win_width  = 1920;
    int          win_height = 1080;
    PointsParams pts        = {};
    int          iter_count = 1000000;
    bool         vsync      = true;  // real display vsync (DVZ_CANVAS_FLAGS_VSYNC)
    bool         display    = true;
    float        size_px    = 5.f;   // marker size in pixels (same as benchmark_mimir --size)
    bool         sphere3d   = false; // lit sphere impostors instead of flat discs
};

// Map --light-model none|phong|path-tracing onto datoviz's raster geometry.
// datoviz is the raster baseline (DESIGN_pathtracing.md §7): 'none' -> flat discs,
// 'phong' -> lit sphere impostors, 'path-tracing' is unavailable and exits.
static bool parseLightModelSphere(const std::string& v)
{
    if (v == "none")  return false;
    if (v == "phong") return true;
    if (v == "path-tracing")
    {
        fprintf(stderr,
            "datoviz is the raster baseline and cannot path trace; "
            "run benchmark_mimir --light-model path-tracing instead.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Unknown --light-model '%s' (use none|phong|path-tracing)\n", v.c_str());
    exit(EXIT_FAILURE);
}

struct HudData {
    unsigned int points;
    int          frame;
    float        fps;
    float        compute_ms;
    float        pack_ms;      // always 0 — positions are already packed float3
    float        d2h_ms;       // cudaMemcpy Device->Host (PCIe DMA)
    float        h2h_ms;       // dvz_marker_position (pinned host -> datoviz internal heap copy)
    float        render_ms;    // dvz_scene_step: H2D + D2D + draw (inseparable)
    float        gpu_watts;
    char         gpu_name[256];
    char         gpu_device[64];   // "N (CC major.minor)"
    float        gpu_total_gb;
    float        buf_mb;
    uint32_t     seed;
    unsigned int k;
    float        epsilon;
};

// Mirrors mimir's PerformanceMetrics so CSV columns line up.
// pipeline is not measurable via datoviz and is always 0.
// devmem usage/budget are substituted with NVML used/total.
// graphics_time = dvz_scene_step wall-clock (H2D staging write + D2D upload + draw).
// transfer.{pack,d2h,h2h} are the measurable stages before dvz_scene_step.
struct PerformanceMetrics {
    float frame_rate;
    struct { float compute; float graphics; float pipeline; } times;
    struct { float usage; float budget; } devmem;
    struct { float pack; float d2h; float h2h; } transfer;
};

struct GPUMemoryMetrics { double free, reserved, total, used; };

struct BenchmarkResult {
    PerformanceMetrics perf;
    GPUPowerMetrics    power;
    GPUMemoryMetrics   memory;
};

struct DatovizContext {
    DvzApp*     app;
    DvzBatch*   batch;
    DvzScene*   scene;
    DvzFigure*  figure;
    DvzPanel*   panel;
    DvzArcball* arcball;
    DvzVisual*  visual;
};

// ---------------------------------------------------------------------------
// Output formatting helpers
// ---------------------------------------------------------------------------

static std::string sf(float v)  { char b[32]; snprintf(b, sizeof(b), "%f", v); return b; }
static std::string sd(int v)    { return std::to_string(v); }
static std::string su(uint32_t v) { return std::to_string(v); }
static std::string smb(size_t bytes) {
    char b[32]; snprintf(b, sizeof(b), "%.2f MB", bytes / (1024.0 * 1024.0)); return b;
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

static void printSystemInfo(PointsInput input, size_t rng_bytes, size_t cluster_bytes)
{
    int device_id = -1;
    checkCuda(cudaGetDevice(&device_id));
    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, device_id));

    char gpu_name[256] = "Unknown";
    nvmlDeviceGetName(getNvmlDevice(), gpu_name, sizeof(gpu_name));
    nvmlMemory_v2_t mi;
    mi.version = (unsigned int)(sizeof(mi) | (2 << 24U));
    nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
    constexpr double gb = 1024.0 * 1024.0 * 1024.0;
    size_t pos_bytes = sizeof(float) * 3 * input.pts.count;
    fprintf(stderr, "GPU: %s\n", gpu_name);
    fprintf(stderr, "CUDA device: %d (%s, CC %d.%d)\n",
        device_id, prop.name, prop.major, prop.minor);
    fprintf(stderr, "Total GPU memory: %.2f GB\n", mi.total / gb);
    fprintf(stderr, "Points: %u  (seed %u)\n", input.pts.count, input.pts.seed);
    fprintf(stderr, "Init distribution: %u modes, epsilon %.4f\n", input.pts.k, input.pts.epsilon);
    fprintf(stderr, "Buffers:\n");
    fprintf(stderr, "  positions  (CUDA):     %s\n", smb(pos_bytes).c_str());
    fprintf(stderr, "  positions  (pinned):   %s\n", smb(pos_bytes).c_str());
    fprintf(stderr, "  rng states (CUDA):     %s\n", smb(rng_bytes).c_str());
    fprintf(stderr, "  clusters   (CUDA):     %s\n", smb(cluster_bytes).c_str());
    fprintf(stderr, "  Total:                 %s\n",
        smb(2 * pos_bytes + rng_bytes + cluster_bytes).c_str());
}

void formatResults(PointsInput input, BenchmarkResult result)
{
    std::string mode = input.display ? "datoviz" : "none";
    std::string resolution = "None";
    if      (input.win_width == 1920 && input.win_height == 1080) resolution = "FHD";
    else if (input.win_width == 2560 && input.win_height == 1440) resolution = "QHD";
    else if (input.win_width == 3840 && input.win_height == 2160) resolution = "UHD";

    auto lib  = result.perf;
    auto gpu  = result.power;
    auto nvml = result.memory;

    // Same column order as benchmark_mimir.
    // graphics_time = dvz_scene_step (H2D staging write + D2D upload + draw, inseparable).
    // pack_time is 0 (positions already packed); d2h_time/h2h_time are the transfer stages.
    printAligned({
        {"mode",          mode},
        {"windowres",     resolution},
        {"N",             sd((int)input.pts.count)},
        {"seed",          su(input.pts.seed)},
        {"k",             su(input.pts.k)},
        {"epsilon",       sf(input.pts.epsilon)},
        {"framerate",     sf(lib.frame_rate)},
        {"compute_time",  sf(lib.times.compute)},
        {"pipeline_time", sf(lib.times.pipeline)},
        {"graphics_time", sf(lib.times.graphics)},
        {"vk_usage",      sf(lib.devmem.usage)},
        {"vk_budget",     sf(lib.devmem.budget)},
        {"gpu_power",     sf(gpu.average_power)},
        {"gpu_energy",    sf(gpu.total_energy)},
        {"gpu_time",      sf(gpu.total_time)},
        {"nvml_free",     sf((float)nvml.free)},
        {"nvml_reserved", sf((float)nvml.reserved)},
        {"nvml_total",    sf((float)nvml.total)},
        {"nvml_used",     sf((float)nvml.used)},
        {"pack_time",     sf(lib.transfer.pack)},
        {"d2h_time",      sf(lib.transfer.d2h)},
        {"h2h_time",      sf(lib.transfer.h2h)},
    });
    printf("%s,%s,%u,%u,%u,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(), resolution.c_str(),
        input.pts.count, input.pts.seed, input.pts.k, input.pts.epsilon,
        lib.frame_rate, lib.times.compute, lib.times.pipeline, lib.times.graphics,
        lib.devmem.usage, lib.devmem.budget,
        gpu.average_power, gpu.total_energy, gpu.total_time,
        nvml.free, nvml.reserved, nvml.total, nvml.used,
        lib.transfer.pack, lib.transfer.d2h, lib.transfer.h2h);
}

// ---------------------------------------------------------------------------
// Keyboard callback — Ctrl+W closes the window
// ---------------------------------------------------------------------------

static void keyCallback(DvzApp* /*app*/, DvzId /*window_id*/, DvzKeyboardEvent* ev)
{
    if (ev->type == DVZ_KEYBOARD_EVENT_PRESS
        && ev->key  == DVZ_KEY_W
        && (ev->mods & DVZ_KEY_MODIFIER_CONTROL))
    {
        auto* flag = static_cast<std::atomic<bool>*>(ev->user_data);
        flag->store(true);
    }
}

// ---------------------------------------------------------------------------
// HUD callback (datoviz GUI — called from within dvz_scene_step)
// ---------------------------------------------------------------------------

static void hudCallback(DvzApp* /*app*/, DvzId /*canvas_id*/, DvzGuiEvent* ev)
{
    auto* hud = static_cast<HudData*>(ev->user_data);
    dvz_gui_pos((vec2){10, 10}, (vec2){0, 0});
    dvz_gui_begin("Performance", 0);
    dvz_gui_text("GPU        %s",       hud->gpu_name);
    dvz_gui_text("Device     %s",       hud->gpu_device);
    dvz_gui_text("VRAM       %.1f GB",  hud->gpu_total_gb);
    dvz_gui_text("Points     %u",       hud->points);
    dvz_gui_text("Seed       %u",       hud->seed);
    dvz_gui_text("Modes k    %u",       hud->k);
    dvz_gui_text("Epsilon    %.4f",     hud->epsilon);
    dvz_gui_text("Buffers    %.1f MB",  hud->buf_mb);
    dvz_gui_text("");
    dvz_gui_text("Frame      %d",        hud->frame);
    dvz_gui_text("FPS        %.1f",      hud->fps);
    dvz_gui_text("Compute    %.2f ms",   hud->compute_ms);
    dvz_gui_text("Transfer   %.2f ms",   hud->pack_ms + hud->d2h_ms + hud->h2h_ms);
    dvz_gui_text("    Pack   %.2f ms",   hud->pack_ms);
    dvz_gui_text("    D2H    %.2f ms",   hud->d2h_ms);
    dvz_gui_text("    H2H    %.2f ms",   hud->h2h_ms);
    dvz_gui_text("Render     %.2f ms (H2D + D2D + draw)", hud->render_ms);
    dvz_gui_text("Power      %.1f W",   hud->gpu_watts);
    dvz_gui_end();
}

// ---------------------------------------------------------------------------
// datoviz setup
// ---------------------------------------------------------------------------

static DatovizContext setupDatoviz(PointsInput input, const float* initial_pos3, unsigned int n)
{
    DatovizContext ctx{};
    ctx.app   = dvz_app(DVZ_APP_FLAGS_NONE);
    ctx.batch = dvz_app_batch(ctx.app);
    ctx.scene = dvz_scene(ctx.batch);

    int fig_flags = DVZ_CANVAS_FLAGS_IMGUI;
    if (input.vsync) fig_flags |= DVZ_CANVAS_FLAGS_VSYNC;
    ctx.figure  = dvz_figure(ctx.scene, input.win_width, input.win_height, fig_flags);
    ctx.panel   = dvz_panel_default(ctx.figure);
    ctx.arcball = dvz_panel_arcball(ctx.panel, 0); // 3D interactivity

    std::vector<DvzColor> colors(n);
    for (unsigned int i = 0; i < n; i++)
    {
        colors[i][0] = 255; colors[i][1] = 255; colors[i][2] = 255; colors[i][3] = 255;
    }

    if (input.sphere3d)
    {
        // Lit sphere impostors (ray-sphere point sprites with Phong lighting and
        // per-fragment depth) — same technique as mimir's Sphere3D marker mode.
        // dvz_sphere_size is the sphere diameter in NDC units; mimir's Sphere3D
        // uses radius = size/100 world units, so size/50 gives the same geometry
        // (the [-1,1]^3 domain maps 1:1 to NDC under the arcball).
        ctx.visual = dvz_sphere(ctx.batch, DVZ_SPHERE_FLAGS_LIGHTING);
        dvz_sphere_alloc(ctx.visual, n);
        // Directional sun (w=0 -> lighting.glsl normalizes light.pos and ignores
        // distance) coming diagonally from behind-and-above the camera. Same world-space
        // convention and SAME normalized vector as benchmark_mimir's opts.light_pos, so
        // both samples are lit by one shared sun instead of datoviz's default light.
        vec4 sun_dir = { -0.4082f, 0.4082f, 0.8165f, 0.f }; // normalize({-1, 1, 2}), w=0
        dvz_sphere_light_pos(ctx.visual, 0, sun_dir);
        dvz_sphere_position(ctx.visual, 0, n, (vec3*)initial_pos3, 0);
        std::vector<float> sizes(n, input.size_px / 50.f);
        dvz_sphere_size(ctx.visual, 0, n, sizes.data(), 0);
        dvz_sphere_color(ctx.visual, 0, n, colors.data(), 0);
    }
    else
    {
        // Filled disc markers sized in pixels — matches mimir's Flat2D Markers view.
        ctx.visual = dvz_marker(ctx.batch, 0);
        dvz_marker_mode(ctx.visual, DVZ_MARKER_MODE_CODE);
        dvz_marker_aspect(ctx.visual, DVZ_MARKER_ASPECT_FILLED);
        dvz_marker_shape(ctx.visual, DVZ_MARKER_SHAPE_DISC);
        dvz_marker_alloc(ctx.visual, n);
        dvz_marker_position(ctx.visual, 0, n, (vec3*)initial_pos3, 0);
        std::vector<float> sizes(n, input.size_px);
        dvz_marker_size(ctx.visual, 0, n, sizes.data(), 0);
        dvz_marker_color(ctx.visual, 0, n, colors.data(), 0);
    }

    dvz_panel_visual(ctx.panel, ctx.visual, 0);
    return ctx;
}

// ---------------------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------------------

BenchmarkResult runExperiment(PointsInput input)
{
    const size_t n         = input.pts.count;
    const size_t pos_bytes = sizeof(float) * 3 * n;

    checkCuda(cudaSetDevice(0));

    // Plain (non-interop) device buffer: this is the whole point of the comparison.
    float* d_pos = nullptr;
    checkCuda(cudaMalloc((void**)&d_pos, pos_bytes));

    // Pinned host buffer for fast D2H — same float3 layout as d_pos.
    float* h_pos = nullptr;
    checkCuda(cudaHostAlloc((void**)&h_pos, pos_bytes, cudaHostAllocDefault));

    auto rng      = createRngStates(input.pts.seed);
    auto clusters = createClusters(input.pts);
    launchInitPositions(d_pos, input.pts, clusters, rng);

    // Prime the pinned buffer for datoviz visual creation.
    checkCuda(cudaMemcpy(h_pos, d_pos, pos_bytes, cudaMemcpyDeviceToHost));

    std::atomic<bool> quit_flag{false};

    DatovizContext ctx{};
    HudData hud{};
    hud.points  = (unsigned int)n;
    hud.seed    = input.pts.seed;
    hud.k       = input.pts.k;
    hud.epsilon = input.pts.epsilon;

    if (input.display)
    {
        ctx = setupDatoviz(input, h_pos, (unsigned int)n);
        dvz_app_gui(ctx.app, dvz_figure_id(ctx.figure), hudCallback, &hud);
        dvz_app_on_keyboard(ctx.app, keyCallback, &quit_flag);

        // GPU name/VRAM need NVML, which is only initialized later by GPUPowerBegin(); they are
        // filled in after that call. buf_mb and the CUDA device info need no NVML.
        hud.buf_mb = (float)((2 * pos_bytes + rngStatesBytes(rng) + clusterBytes(clusters, n))
            / (1024.0 * 1024.0));
        int device_id = -1;
        checkCuda(cudaGetDevice(&device_id));
        cudaDeviceProp prop{};
        checkCuda(cudaGetDeviceProperties(&prop, device_id));
        snprintf(hud.gpu_device, sizeof(hud.gpu_device), "%d (CC %d.%d)",
            device_id, prop.major, prop.minor);
    }

    // CUDA events for integrate kernel timing.
    cudaEvent_t cstart, cstop;
    checkCuda(cudaEventCreate(&cstart));
    checkCuda(cudaEventCreate(&cstop));

    float total_compute  = 0.f;
    float total_d2h      = 0.f;
    float total_h2h      = 0.f;
    float total_graphics = 0.f;
    size_t frame_count   = 0;

    GPUPowerBegin("gpu", 100);
    printSystemInfo(input, rngStatesBytes(rng), clusterBytes(clusters, n));

    if (input.display)
    {
        // NVML is initialized by GPUPowerBegin() above, so the GPU name/VRAM queries succeed
        // here (they returned 0 when issued before init).
        nvmlDeviceGetName(getNvmlDevice(), hud.gpu_name, sizeof(hud.gpu_name));
        nvmlMemory_v2_t mi;
        mi.version = (unsigned int)(sizeof(mi) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
        hud.gpu_total_gb = (float)(mi.total / (1024.0 * 1024.0 * 1024.0));
    }

    auto loop_start = clk::now();

    for (int i = 0; i < input.iter_count; ++i)
    {
        // --- Compute (random-walk step only) ---
        // NOTE: windowed, this reads ~0.1 ms higher than mimir's ~0.02 ms for the SAME
        // kernel. The kernel really is ~0.02 ms (see --display 0, which drops it to that);
        // the extra ~0.1 ms is the GPU graphics->compute transition latency datoviz pays
        // because it separates render from compute with a full vkDeviceWaitIdle
        // (dvz_app_wait, below) rather than mimir's fine-grained interop timeline
        // semaphore. It is a genuine per-frame cost of the non-interop path, not the
        // kernel, and is ~0.2% of the render-bound frame.
        checkCuda(cudaEventRecord(cstart));
        launchIntegrate3D(d_pos, n, clusters, rng);
        checkCuda(cudaEventRecord(cstop));
        checkCuda(cudaEventSynchronize(cstop));
        float compute_ms = 0.f;
        checkCuda(cudaEventElapsedTime(&compute_ms, cstart, cstop));
        total_compute += compute_ms;

        if (input.display)
        {
            // --- D2H: float3 positions (device) -> pinned host (PCIe DMA) ---
            auto t_d2h = clk::now();
            checkCuda(cudaMemcpy(h_pos, d_pos, pos_bytes, cudaMemcpyDeviceToHost));
            float d2h_ms = (float)ms_since(t_d2h);

            // --- H2H: pinned host -> datoviz's internal heap copy (CPU memcpy) ---
            auto t_h2h = clk::now();
            if (input.sphere3d) dvz_sphere_position(ctx.visual, 0, (uint32_t)n, (vec3*)h_pos, 0);
            else                dvz_marker_position(ctx.visual, 0, (uint32_t)n, (vec3*)h_pos, 0);
            float h2h_ms = (float)ms_since(t_h2h);

            total_d2h += d2h_ms;
            total_h2h += h2h_ms;

            // --- Render: dvz_scene_step = H2D staging write + D2D upload + draw ---
            // These GPU/CPU operations are inseparable via the datoviz public API.
            // dvz_scene_step submits the frame ASYNCHRONOUSLY and returns after only the
            // CPU-side work, so without a fence the real GPU render cost leaks into the
            // NEXT frame's compute timing (the CUDA kernel serializes behind the still-
            // running Vulkan render on the shared GPU, inflating "compute" from ~0.02 ms
            // to ~16 ms). dvz_app_wait() drains the GPU here so graphics_ms is the true
            // render cost and the next kernel is measured uncontended. This serializes
            // compute vs render exactly like mimir's interop lockstep, giving an
            // apples-to-apples per-phase comparison (at the cost of datoviz's async
            // CPU/GPU pipelining, which would otherwise raise its throughput).
            auto t_render = clk::now();
            if (!dvz_scene_step(ctx.scene, ctx.app) || quit_flag) break;
            dvz_app_wait(ctx.app);
            float graphics_ms = (float)ms_since(t_render);
            total_graphics += graphics_ms;
            ++frame_count;

            float frame_total = compute_ms + d2h_ms + h2h_ms + graphics_ms;
            hud.frame      = (int)frame_count;
            hud.compute_ms = compute_ms;
            hud.pack_ms    = 0.f;
            hud.d2h_ms     = d2h_ms;
            hud.h2h_ms     = h2h_ms;
            hud.render_ms  = graphics_ms;
            if (frame_total > 0.f) {
                float new_fps = 1000.f / frame_total;
                hud.fps = (frame_count == 1) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
            }
            float watts   = (float)getGPUCurrentPower();
            hud.gpu_watts = (frame_count == 1) ? watts : 0.9f * hud.gpu_watts + 0.1f * watts;
        }
    }

    if (!input.display) checkCuda(cudaDeviceSynchronize());

    auto loop_wall = std::chrono::duration<double>(clk::now() - loop_start).count();
    float frame_rate = 0.f;
    if (input.display && loop_wall > 0.0) frame_rate = (float)(frame_count / loop_wall);
    else if (loop_wall > 0.0)             frame_rate = (float)(input.iter_count / loop_wall);

    nvmlMemory_v2_t meminfo;
    meminfo.version = (unsigned int)(sizeof(meminfo) | (2 << 24U));
    nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);
    constexpr double gb = 1024.0 * 1024.0 * 1024.0;
    GPUMemoryMetrics nvml{
        .free     = meminfo.free / gb,
        .reserved = meminfo.reserved / gb,
        .total    = meminfo.total / gb,
        .used     = meminfo.used / gb,
    };

    auto gpu_power = GPUPowerEnd();

    PerformanceMetrics metrics{};
    metrics.frame_rate     = frame_rate;
    metrics.times.compute  = total_compute  / 1000.f;   // ms → s
    metrics.times.graphics = total_graphics / 1000.f;
    metrics.times.pipeline = 0.f;
    metrics.devmem.usage   = (float)nvml.used;
    metrics.devmem.budget  = (float)nvml.total;
    metrics.transfer.pack  = 0.f;
    metrics.transfer.d2h   = total_d2h / 1000.f;
    metrics.transfer.h2h   = total_h2h / 1000.f;

    // Cleanup
    checkCuda(cudaEventDestroy(cstart));
    checkCuda(cudaEventDestroy(cstop));

    if (input.display)
    {
        dvz_scene_destroy(ctx.scene);
        dvz_app_destroy(ctx.app);
    }
    destroyClusters(clusters);
    destroyRngStates(rng);
    checkCuda(cudaFree(d_pos));
    checkCuda(cudaFreeHost(h_pos));

    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(const char* prog)
{
    printf(
        "Usage: %s [win_w win_h] [points] [seed] [iters] [options]\n"
        "\n"
        "Positional (in order; win_w/win_h must be supplied together):\n"
        "  win_w  win_h   Window resolution in pixels             (default: 1920 1080)\n"
        "  points         Number of simulated points              (default: 1000000)\n"
        "  seed           RNG seed for positions/walk             (default: 12345)\n"
        "  iters          Simulation steps to run                 (default: 1000000)\n"
        "\n"
        "Options (named, order-independent; omitted ones use their default):\n"
        "  --vsync N          display vsync: 1=on 0=off            (default: 1)\n"
        "  --display N        1 = open window, 0 = headless compute (default: 1)\n"
        "  --size S           Marker size in pixels                 (default: 5)\n"
        "                     Same meaning as benchmark_mimir --size.\n"
        "  --light-model M    none         = flat disc markers (dvz_marker),\n"
        "                     phong        = lit sphere impostors (dvz_sphere),\n"
        "                     path-tracing = unavailable (datoviz is the raster\n"
        "                                    baseline; exits)      (default: none)\n"
        "                     'phong' matches benchmark_mimir --light-model phong: the\n"
        "                     sphere radius is size/100 in [-1,1] domain units.\n"
        "  --k N              Gaussian modes (clusters) at init     (default: 8)\n"
        "  --epsilon E        Per-axis stddev of each mode          (default: 0.05)\n"
        "                     The walk is mean-reverting, so clusters keep this\n"
        "                     stddev over time. Centers are seed-deterministic;\n"
        "                     same cloud as benchmark_mimir for equal args.\n"
        "\n"
        "(datoviz has no present-mode selection; the mimir --present flag has no\n"
        " datoviz equivalent, so it is intentionally absent here.)\n"
        "Rendering GPU: defaults to Vulkan device 0 to match the CUDA device; if your\n"
        "        CUDA and Vulkan device orders differ, set DVZ_GPU=<idx> to the Vulkan\n"
        "        index of the CUDA GPU.\n"
        "Frame rate is always uncapped (no target_fps limiter).\n"
        "Output: one CSV row to stdout.\n"
        "        Columns match benchmark_mimir; pack_time is 0 (positions are already\n"
        "        packed float3), d2h_time/h2h_time are the transfer stages.\n"
        "        graphics_time = dvz_scene_step (H2D staging write + D2D upload + draw,\n"
        "        inseparable).\n",
        prog);
}

int main(int argc, char* argv[])
{
    if (argc == 1) { usage(argv[0]); return EXIT_SUCCESS; }

    PointsInput input{};
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
            if      (a == "--vsync")       input.vsync    = (bool)std::stoi(v);
            else if (a == "--display")     input.display  = (bool)std::stoi(v);
            else if (a == "--size")        input.size_px  = std::stof(v);
            else if (a == "--light-model") input.sphere3d = parseLightModelSphere(v);
            else if (a == "--k")           input.pts.k = (unsigned int)std::stoul(v);
            else if (a == "--epsilon")     input.pts.epsilon = std::stof(v);
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { pos.push_back(a); }
    }
    if (pos.size() >= 2) { input.win_width = std::stoi(pos[0]); input.win_height = std::stoi(pos[1]); }
    if (pos.size() >= 3)   input.pts.count = (unsigned int)std::stoul(pos[2]);
    if (pos.size() >= 4)   input.pts.seed  = (uint32_t)std::stoul(pos[3]);
    if (pos.size() >= 5)   input.iter_count = std::stoi(pos[4]);

    // CUDA runs on device 0 (cudaSetDevice in runExperiment); render on the same GPU
    // instead of datoviz's own "best GPU" pick, which can land on a different card in
    // multi-GPU systems (the benchmark would then measure cross-GPU traffic, and the
    // pick may lack a swapchain). Vulkan and CUDA enumeration order can still differ;
    // set DVZ_GPU=<idx> explicitly if they do.
    setenv("DVZ_GPU", "0", /*overwrite=*/0);

    auto result = runExperiment(input);
    formatResults(input, result);
    return EXIT_SUCCESS;
}
