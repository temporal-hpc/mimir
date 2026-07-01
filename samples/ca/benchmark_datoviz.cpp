// 2D Game of Life benchmark — datoviz rendering path.
//
// Identical simulation to benchmark_mimir.cpp (same kernel, same initGrid seed).
// The only difference is the rendering path:
//
//   mimir   : CUDA writes R8 pixels into a Vulkan/CUDA shared buffer — zero transfer.
//   datoviz : has no CUDA interop; each frame the pixel buffer must round-trip:
//               1. D2H    uint8 grid (device) -> pinned host buffer     (PCIe DMA)
//               2. H2H    pinned host -> datoviz's internal heap copy   (CPU memcpy,
//                                                                        inside dvz_texture_data)
//               3. H2D    heap copy -> Vulkan staging VkBuffer          (CPU memcpy,
//                                                                        inside dvz_scene_step)
//               4. D2D    staging VkBuffer -> device-local VkImage      (GPU copy + tiling,
//                                                                        inside dvz_scene_step)
//               5. draw   render the fullscreen quad                    (GPU, inside dvz_scene_step)
//
// The grid is already single-byte grayscale (0/255 per cell), so no RGBA packing
// kernel is needed: the texture is DVZ_FORMAT_R8_UNORM, displayed via datoviz's
// built-in DVZ_CMAP_BINARY colormap (DVZ_IMAGE_FLAGS_MODE_COLORMAP). This is a 4x
// reduction in bytes moved per frame versus an RGBA8 texture.
//
// Steps 3, 4 and 5 are inseparable via the datoviz public API; all three are
// captured in render_ms (dvz_scene_step wall-clock) and labeled "H2D + D2D + draw"
// in the HUD and CSV output.

#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include <datoviz.h>

#include "ca_sim.cuh"
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

struct CAInput {
    int      win_width   = 1920;
    int      win_height  = 1080;
    CAParams ca          = {};
    int      iter_count  = 1000000;
    bool     vsync       = true;  // real display vsync (DVZ_CANVAS_FLAGS_VSYNC)
    bool     display     = true;
};

struct HudData {
    unsigned int cells;
    int          frame;
    float        fps;
    float        compute_ms;
    float        pack_ms;      // always 0 — grid is already single-byte grayscale
    float        d2h_ms;       // cudaMemcpy Device->Host (PCIe DMA)
    float        h2h_ms;       // dvz_texture_data (pinned host -> datoviz internal heap copy)
    float        render_ms;    // dvz_scene_step: H2D + D2D + draw (inseparable)
    float        gpu_watts;
    char         gpu_name[256];
    float        gpu_total_gb;
    float        buf_mb;
    int          grid_w, grid_h;
    float        density;
    uint32_t     seed;
};

// Mirrors mimir's PerformanceMetrics so CSV columns line up.
// pipeline is not measurable via datoviz and is always 0.
// devmem usage/budget are substituted with NVML used/total.
// graphics_time = dvz_scene_step wall-clock (H2D staging write + D2D retile + draw).
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
    DvzVisual*  visual;
    DvzTexture* texture;
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

static void printSystemInfo(CAInput input)
{
    char gpu_name[256] = "Unknown";
    nvmlDeviceGetName(getNvmlDevice(), gpu_name, sizeof(gpu_name));
    nvmlMemory_v2_t mi;
    mi.version = (unsigned int)(sizeof(mi) | (2 << 24U));
    nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
    constexpr double gb = 1024.0 * 1024.0 * 1024.0;
    size_t N = (size_t)input.ca.width * input.ca.height;
    fprintf(stderr, "GPU: %s\n", gpu_name);
    fprintf(stderr, "Total GPU memory: %.2f GB\n", mi.total / gb);
    fprintf(stderr, "Grid: %d x %d  (%.2f M cells)\n", input.ca.width, input.ca.height, N / 1e6);
    fprintf(stderr, "Buffers:\n");
    fprintf(stderr, "  d_grid[0]  (CUDA):    %s\n", smb(N).c_str());
    fprintf(stderr, "  d_grid[1]  (CUDA):    %s\n", smb(N).c_str());
    fprintf(stderr, "  h_pixels   (pinned):  %s\n", smb(N).c_str());
    fprintf(stderr, "  Total:                %s\n", smb(N * 3).c_str());
}

void formatResults(CAInput input, BenchmarkResult result)
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
    // graphics_time = dvz_scene_step (H2D staging write + D2D retile + draw, inseparable).
    // pack_time/d2h_time/h2h_time are the measurable transfer stages.
    printAligned({
        {"mode",          mode},
        {"windowres",     resolution},
        {"grid_w",        sd(input.ca.width)},
        {"grid_h",        sd(input.ca.height)},
        {"seed",          su(input.ca.seed)},
        {"density",       sf(input.ca.density)},
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
    printf("%s,%s,%d,%d,%u,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(), resolution.c_str(),
        input.ca.width, input.ca.height, input.ca.seed, input.ca.density,
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
    dvz_gui_text("VRAM       %.1f GB",  hud->gpu_total_gb);
    dvz_gui_text("Grid       %d x %d",  hud->grid_w, hud->grid_h);
    dvz_gui_text("Seed       %u",       hud->seed);
    dvz_gui_text("Density    %.2f",     hud->density);
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

static DatovizContext setupDatoviz(CAInput input, const uint8_t* initial_pixels, int W, int H)
{
    DatovizContext ctx{};
    ctx.app   = dvz_app(DVZ_APP_FLAGS_NONE);
    ctx.batch = dvz_app_batch(ctx.app);
    ctx.scene = dvz_scene(ctx.batch);

    int fig_flags = DVZ_CANVAS_FLAGS_IMGUI;
    if (input.vsync) fig_flags |= DVZ_CANVAS_FLAGS_VSYNC;
    ctx.figure = dvz_figure(ctx.scene, input.win_width, input.win_height, fig_flags);
    ctx.panel  = dvz_panel_default(ctx.figure);
    dvz_panel_panzoom(ctx.panel, 0);

    // Single-channel R8 texture displayed through datoviz's built-in grayscale
    // colormap — no RGBA packing kernel needed.
    ctx.visual = dvz_image(ctx.batch, DVZ_IMAGE_FLAGS_MODE_COLORMAP);
    dvz_image_colormap(ctx.visual, DVZ_CMAP_BINARY);
    dvz_image_alloc(ctx.visual, 1);

    // Center the image in the panel, scaled to fill the window.
    vec3 pos    = {0, 0, 0};
    vec2 size   = {(float)input.win_width, (float)input.win_height};
    vec2 anchor = {0, 0};          // anchor (0,0) = center of image at position
    vec4 uv     = {0, 0, 1, 1};   // full texture
    dvz_image_position(ctx.visual, 0, 1, &pos,    0);
    dvz_image_size    (ctx.visual, 0, 1, &size,   0);
    dvz_image_anchor  (ctx.visual, 0, 1, &anchor, 0);
    dvz_image_texcoords(ctx.visual, 0, 1, &uv,    0);

    ctx.texture = dvz_texture_2D(
        ctx.batch,
        DVZ_FORMAT_R8_UNORM,
        DVZ_FILTER_NEAREST,
        DVZ_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        (uint32_t)W, (uint32_t)H,
        (void*)initial_pixels, 0);
    dvz_image_texture(ctx.visual, ctx.texture);
    dvz_panel_visual(ctx.panel, ctx.visual, 0);
    return ctx;
}

// ---------------------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------------------

BenchmarkResult runExperiment(CAInput input)
{
    const int    W = input.ca.width;
    const int    H = input.ca.height;
    const size_t N = (size_t)W * H;

    checkCuda(cudaSetDevice(0));

    // Two ping-pong CA grid buffers (uint8, plain CUDA).
    uint8_t* d_grid[2] = {};
    checkCuda(cudaMalloc(&d_grid[0], N));
    checkCuda(cudaMalloc(&d_grid[1], N));
    int r = 0, w = 1;

    // Pinned host buffer for fast D2H — same byte layout as d_grid (0/255 per cell).
    uint8_t* h_pixels = nullptr;
    checkCuda(cudaHostAlloc((void**)&h_pixels, N, cudaHostAllocDefault));

    auto h_grid = initGrid(input.ca);
    checkCuda(cudaMemcpy(d_grid[r], h_grid.data(), N, cudaMemcpyHostToDevice));

    // Prime the pinned buffer for datoviz texture creation.
    checkCuda(cudaMemcpy(h_pixels, d_grid[r], N, cudaMemcpyDeviceToHost));

    std::atomic<bool> quit_flag{false};

    DatovizContext ctx{};
    HudData hud{};
    hud.cells   = (unsigned int)N;
    hud.grid_w  = W;
    hud.grid_h  = H;
    hud.density = input.ca.density;
    hud.seed    = input.ca.seed;

    if (input.display)
    {
        ctx = setupDatoviz(input, h_pixels, W, H);
        dvz_app_gui(ctx.app, dvz_figure_id(ctx.figure), hudCallback, &hud);
        dvz_app_on_keyboard(ctx.app, keyCallback, &quit_flag);

        // GPU name/VRAM need NVML, which is only initialized later by GPUPowerBegin(); they are
        // filled in after that call. buf_mb needs no NVML, so set it here.
        // 2 x uint8 grid + pinned uint8 buffer
        hud.buf_mb = (float)((2 * N + N) / (1024.0 * 1024.0));
    }

    // CUDA events for GoL compute timing.
    cudaEvent_t cstart, cstop;
    checkCuda(cudaEventCreate(&cstart));
    checkCuda(cudaEventCreate(&cstop));

    float total_compute  = 0.f;
    float total_d2h      = 0.f;
    float total_h2h      = 0.f;
    float total_graphics = 0.f;
    size_t frame_count   = 0;

    GPUPowerBegin("gpu", 100);
    printSystemInfo(input);

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
        // --- Compute (GoL step only) ---
        checkCuda(cudaEventRecord(cstart));
        launchStepGoL(d_grid[r], d_grid[w], W, H);
        checkCuda(cudaEventRecord(cstop));
        checkCuda(cudaEventSynchronize(cstop));
        float compute_ms = 0.f;
        checkCuda(cudaEventElapsedTime(&compute_ms, cstart, cstop));
        total_compute += compute_ms;

        std::swap(r, w);

        if (input.display)
        {
            // --- D2H: uint8 grid (device) -> pinned host (PCIe DMA) ---
            auto t_d2h = clk::now();
            checkCuda(cudaMemcpy(h_pixels, d_grid[r], N, cudaMemcpyDeviceToHost));
            float d2h_ms = (float)ms_since(t_d2h);

            // --- H2H: pinned host -> datoviz's internal heap copy (CPU memcpy) ---
            auto t_h2h = clk::now();
            dvz_texture_data(ctx.texture, 0, 0, 0,
                (uint32_t)W, (uint32_t)H, 1,
                (DvzSize)N, h_pixels);
            float h2h_ms = (float)ms_since(t_h2h);

            total_d2h += d2h_ms;
            total_h2h += h2h_ms;

            // --- Render: dvz_scene_step = H2D staging write + D2D retile + draw ---
            // These GPU/CPU operations are inseparable via the datoviz public API.
            auto t_render = clk::now();
            if (!dvz_scene_step(ctx.scene, ctx.app) || quit_flag) break;
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
    metrics.frame_rate          = frame_rate;
    metrics.times.compute       = total_compute  / 1000.f;   // ms → s
    metrics.times.graphics      = total_graphics / 1000.f;
    metrics.times.pipeline      = 0.f;
    metrics.devmem.usage        = (float)nvml.used;
    metrics.devmem.budget       = (float)nvml.total;
    metrics.transfer.pack       = 0.f;
    metrics.transfer.d2h        = total_d2h / 1000.f;
    metrics.transfer.h2h        = total_h2h / 1000.f;

    // Cleanup
    checkCuda(cudaEventDestroy(cstart));
    checkCuda(cudaEventDestroy(cstop));

    if (input.display)
    {
        dvz_scene_destroy(ctx.scene);
        dvz_app_destroy(ctx.app);
    }
    checkCuda(cudaFree(d_grid[0]));
    checkCuda(cudaFree(d_grid[1]));
    checkCuda(cudaFreeHost(h_pixels));

    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(const char* prog)
{
    printf(
        "Usage: %s [win_w win_h] [grid_w grid_h] [seed] [density] [iters] [options]\n"
        "\n"
        "Positional (in order; win_w/win_h and grid_w/grid_h must each be a pair):\n"
        "  win_w  win_h   Window resolution in pixels             (default: 1920 1080)\n"
        "  grid_w grid_h  CA grid dimensions in cells             (default: 1024 1024)\n"
        "  seed           RNG seed for initial state              (default: 12345)\n"
        "  density        Initial live-cell fraction [0,1]        (default: 0.30)\n"
        "  iters          Simulation steps to run                 (default: 1000000)\n"
        "\n"
        "Options (named, order-independent; omitted ones use their default):\n"
        "  --vsync N          display vsync: 1=on 0=off            (default: 1)\n"
        "  --display N        1 = open window, 0 = headless compute (default: 1)\n"
        "\n"
        "(datoviz has no present-mode selection; the mimir --present flag has no\n"
        " datoviz equivalent, so it is intentionally absent here.)\n"
        "Frame rate is always uncapped (no target_fps limiter).\n"
        "Output: one CSV row to stdout.\n"
        "        Columns match benchmark_mimir plus pack_time, d2h_time, h2h_time.\n"
        "        graphics_time = dvz_scene_step (H2D staging write + D2D retile + draw,\n"
        "        inseparable).\n",
        prog);
}

int main(int argc, char* argv[])
{
    if (argc == 1) { usage(argv[0]); return EXIT_SUCCESS; }

    CAInput input{};
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
            if      (a == "--vsync")   input.vsync   = (bool)std::stoi(v);
            else if (a == "--display") input.display = (bool)std::stoi(v);
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { pos.push_back(a); }
    }
    if (pos.size() >= 2) { input.win_width  = std::stoi(pos[0]); input.win_height = std::stoi(pos[1]); }
    if (pos.size() >= 4) { input.ca.width   = std::stoi(pos[2]); input.ca.height  = std::stoi(pos[3]); }
    if (pos.size() >= 5)   input.ca.seed    = (uint32_t)std::stoul(pos[4]);
    if (pos.size() >= 6)   input.ca.density = std::stof(pos[5]);
    if (pos.size() >= 7)   input.iter_count = std::stoi(pos[6]);

    auto result = runExperiment(input);
    formatResults(input, result);
    return EXIT_SUCCESS;
}
