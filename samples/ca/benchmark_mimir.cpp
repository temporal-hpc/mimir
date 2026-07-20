// 2D Game of Life benchmark — mimir rendering path.
//
// Both ping-pong uint8 grid buffers are allocated as mimir interop allocations.
// Vulkan reads the just-written buffer directly each frame — no pack step, no
// extra memory, zero per-frame host transfer.
//
// Compare with benchmark_datoviz.cpp which must round-trip GPU -> host -> GPU.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>
#include <string>

#include "ca_sim.cuh"
#include "nvmlPower.hpp"
#include "validation.hpp"

#include <mimir/mimir.hpp>
using namespace mimir;

// ---------------------------------------------------------------------------
// Input / output structs
// ---------------------------------------------------------------------------

struct CAInput {
    int         win_width    = 1920;
    int         win_height   = 1080;
    CAParams    ca           = {};          // grid dimensions, seed, density
    int         iter_count   = 1000000;
    PresentMode present      = PresentMode::Immediate;
    bool        enable_interop_sync  = true;
    bool        display      = true;
};

struct HudData {
    unsigned int cells;
    int          frame;
    float        fps;
    float        compute_ms;
    float        render_ms;
    float        wait_ms;        // CPU blocked on fence + swapchain acquire (GPU/present backpressure)
    float        record_ms;      // CPU command-buffer recording
    float        submit_ms;      // CPU vkQueueSubmit + vkQueuePresentKHR
    float        gpu_ms;         // true end-to-end GPU frame latency (submit -> fence signalled)
    float        gpu_watts;
    char         gpu_name[256];
    char         gpu_device[64];   // "N (CC major.minor)"
    float        gpu_total_gb;  // total VRAM (NVML)
    // VRAM used (NVML) dismembered so the sub-parts sum to it. External/CUDA are anchored on
    // measured NVML checkpoints at startup; Render/Vulkan are computed; Mimir is the remainder.
    float        vram_used_mb;
    float        vram_external_mb;  // measured: NVML used before we touch the GPU (other processes)
    float        cuda_ctx_mb;       // measured: CUDA context reservation (cudaFree(0) checkpoint)
    float        buf_mb;            // computed: our CUDA sim device buffers (2 x uint8 interop grid)
    float        render_mb;         // computed: extra render geometry (0: image view reads the grid)
    float        vulkan_mb;         // computed: Vulkan render targets (swapchain + depth)
    int          grid_w, grid_h;
    float        density;
    uint32_t     seed;
    bool         reduce;        // true when presenting via a resampled display buffer (clipmap)
    int          disp_w, disp_h;// presented image resolution
    int          view_ox, view_oy; // top-left cell of the visible window (pan)
    int          view_vw, view_vh; // visible window size in cells (zoom)
    float        pack_ms;       // resample (grid -> display buffer) kernel time
};

// Format the live HUD as plain text for mimir's built-in overlay (setHudText). All numbers are
// collected in plain CUDA/NVML above; this is pure string formatting -- no ImGui, no GUI code.
// (FPS / render already appear in the built-in overlay, so they are omitted here.)
static std::string formatHud(const HudData& h)
{
    const float mimir_mb = h.vram_used_mb - h.vram_external_mb - h.cuda_ctx_mb
        - h.buf_mb - h.render_mb - h.vulkan_mb;
    char b[1200];
    int n = snprintf(b, sizeof(b),
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
        "Grid      %dx%d\n"
        "Seed %u  density %.3f\n"
        "Frame     %d\n"
        "Compute   %.2f ms\n"
        "  Wait     %.2f ms\n"
        "  Record   %.2f ms\n"
        "  Submit   %.2f ms\n"
        "Power     %.1f W",
        h.gpu_name, h.gpu_device, h.gpu_total_gb, h.vram_used_mb,
        h.vram_external_mb, h.cuda_ctx_mb, h.buf_mb, h.render_mb, h.vulkan_mb, mimir_mb,
        h.grid_w, h.grid_h, h.seed, h.density,
        h.frame, h.compute_ms, h.wait_ms, h.record_ms, h.submit_ms, h.gpu_watts);
    if (h.reduce && n > 0 && n < (int)sizeof(b))
    {
        snprintf(b + n, sizeof(b) - (size_t)n,
            "\nView      %dx%d @ (%d,%d)\nResample  %.2f ms\n[wheel zoom, WASD/arrows pan, R reset]",
            h.view_vw, h.view_vh, h.view_ox, h.view_oy, h.pack_ms);
    }
    return b;
}

struct GPUMemoryMetrics { double free, reserved, total, used; };

struct BenchmarkResult {
    PerformanceMetrics perf;
    GPUPowerMetrics    power;
    GPUMemoryMetrics   memory;
    // Grid->display downsample time (seconds), mimir's on-GPU analog of datoviz's pack step.
    // 0 when the grid fits within maxImageDimension2D and is presented zero-copy.
    float              pack_time = 0.f;
    int                iters = 0;  // iterations executed; divide the time TOTALS by this for per-frame averages
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

// disp_n = presented image cell count; ds = downsample factor (1 = grid presented at native res).
static void printSystemInfo(CAInput input, size_t disp_n, bool reduce, int dw, int dh)
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
    if (reduce)
    {
        // Grid too large or misaligned to present directly: full-res sim buffers are plain CUDA, and
        // two aligned display buffers (dw x dh) are the interop images actually presented/resampled.
        fprintf(stderr, "  d_grid[0]  (cuda):     %s\n", smb(N).c_str());
        fprintf(stderr, "  d_grid[1]  (cuda):     %s\n", smb(N).c_str());
        fprintf(stderr, "  d_disp[0]  (interop):  %s  (%dx%d)\n", smb(disp_n).c_str(), dw, dh);
        fprintf(stderr, "  d_disp[1]  (interop):  %s  (%dx%d)\n", smb(disp_n).c_str(), dw, dh);
        fprintf(stderr, "  Total:                 %s\n", smb(2 * N + 2 * disp_n).c_str());
    }
    else
    {
        fprintf(stderr, "  d_grid[0]  (interop):  %s\n", smb(N).c_str());
        fprintf(stderr, "  d_grid[1]  (interop):  %s\n", smb(N).c_str());
        fprintf(stderr, "  Total:                 %s\n", smb(2 * N).c_str());
    }
}

void formatResults(CAInput input, BenchmarkResult result)
{
    std::string mode = input.display ? "mimir" : "none";
    std::string resolution = "None";
    if      (input.win_width == 1920 && input.win_height == 1080) resolution = "FHD";
    else if (input.win_width == 2560 && input.win_height == 1440) resolution = "QHD";
    else if (input.win_width == 3840 && input.win_height == 2160) resolution = "UHD";

    auto lib  = result.perf;
    auto gpu  = result.power;
    auto nvml = result.memory;

    // pack_time is the grid->display downsample (0 for grids that fit and are presented zero-copy);
    // d2h_time/h2h_time are always 0 for mimir (no host round-trip). graphics_time = mimir's
    // internal render time. Column layout matches benchmark_datoviz for direct CSV comparison.
    // Column names carry units; time columns are TOTALS over the run in seconds (including
    // pipeline_time_s, summed in runExperiment), memory in GB, power in W, energy in J. iters is
    // placed with the experiment settings, right after the grid size (grid_w x grid_h = N).
    printAligned({
        {"mode",            mode},
        {"windowres",       resolution},
        {"grid_w",          sd(input.ca.width)},
        {"grid_h",          sd(input.ca.height)},
        {"iters",           sd(result.iters)},
        {"seed",            su(input.ca.seed)},
        {"density",         sf(input.ca.density)},
        {"framerate_fps",   sf(lib.frame_rate)},
        {"compute_time_s",  sf(lib.times.compute)},
        {"pipeline_time_s", sf(lib.times.pipeline)},
        {"graphics_time_s", sf(lib.times.graphics)},
        {"vk_usage_gb",     sf(lib.devmem.usage)},
        {"vk_budget_gb",    sf(lib.devmem.budget)},
        {"gpu_power_w",     sf(gpu.average_power)},
        {"gpu_energy_j",    sf(gpu.total_energy)},
        {"gpu_time_s",      sf(gpu.total_time)},
        {"nvml_free_gb",    sf((float)nvml.free)},
        {"nvml_reserved_gb",sf((float)nvml.reserved)},
        {"nvml_total_gb",   sf((float)nvml.total)},
        {"nvml_used_gb",    sf((float)nvml.used)},
        {"pack_time_s",     sf(result.pack_time)},
        {"d2h_time_s",      sf(0.f)},
        {"h2h_time_s",      sf(0.f)},
    });
    printf("%s,%s,%d,%d,%d,%u,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(), resolution.c_str(),
        input.ca.width, input.ca.height, result.iters, input.ca.seed, input.ca.density,
        lib.frame_rate, lib.times.compute, lib.times.pipeline, lib.times.graphics,
        lib.devmem.usage, lib.devmem.budget,
        gpu.average_power, gpu.total_energy, gpu.total_time,
        nvml.free, nvml.reserved, nvml.total, nvml.used,
        result.pack_time, 0.f, 0.f);
}

// ---------------------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------------------

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

BenchmarkResult runExperiment(CAInput input)
{
    const int    W = input.ca.width;
    const int    H = input.ca.height;
    const size_t N = (size_t)W * H;

    // VRAM checkpoint #1: before we touch the GPU -- baseline held by other processes.
    const double vram_external = sampleVramUsedMB();

    checkCuda(cudaSetDevice(0));
    // VRAM checkpoint #2: force the primary CUDA context to exist (cudaFree(0)) and measure its
    // reservation on its own, before our buffers or the mimir/Vulkan renderer are created.
    checkCuda(cudaFree(0));
    double cuda_ctx_mb = sampleVramUsedMB() - vram_external;
    if (cuda_ctx_mb < 0.0) cuda_ctx_mb = 0.0;

    // Mimir instance.
    ViewerOptions opts{};
    opts.window.title        = "Mimir - ca";
    opts.window.size         = { input.win_width, input.win_height };
    opts.background_color    = { 0.f, 0.f, 0.f, 1.f };
    opts.present.mode              = input.present;
    opts.present.enable_interop_sync       = input.enable_interop_sync;
    opts.present.enable_fps_limit  = false;  // always uncapped
    opts.show_hud = true; // built-in overlay (F2); metrics pushed via setHudText below (no ImGui here)
    // The Performance HUD (setGuiCallback) draws regardless of show_panel; keep the engine's
    // scene-parameters panel hidden by default (Ctrl+G shows it, F1 toggles ALL GUI windows).
    opts.show_panel          = false;

    InstanceHandle instance = nullptr;
    createInstance(opts, &instance);

    // A grid larger than the device's maxImageDimension2D (like OpenGL's GL_MAX_TEXTURE_SIZE) cannot
    // be presented as one image regardless of VRAM. The simulation is a linear buffer with no such
    // limit, so we keep it full-res and, when it doesn't fit, present a resampled display buffer sized
    // to the window (aspect-preserved). When it fits we alias the grid buffers directly (zero-copy) at
    // ANY width -- the library transparently handles the interop image's row-pitch alignment.
    const FormatDescription r8_fmt{ .kind = FormatKind::UnsignedNormalized, .size = 1, .components = 1 };
    uint32_t img_cap = maxImageDimension2D(instance);
    if (img_cap == 0) img_cap = 16384;  // portable floor if no device is selected
    const int cap = (int)std::min<uint32_t>(img_cap,
        std::max(1, std::max(input.win_width, input.win_height)));

    const bool fits   = (W <= cap && H <= cap);
    const bool reduce = !fits;

    int DW, DH;
    if (reduce)
    {
        DW = std::min(W, cap);
        if (DW > (int)img_cap) DW = (int)img_cap;
        DH = (int)((long)DW * H / W);              // preserve the grid's aspect
        DH = std::clamp(DH, 1, (int)img_cap);
    }
    else { DW = W; DH = H; }
    const size_t DN = (size_t)DW * DH;

    // Full-res ping-pong sim buffers (uint8). When reducing, these are plain CUDA (never imaged)
    // and the presented images are the separate reduced-resolution display buffers below. When the
    // grid fits, the interop grid buffers ARE the presented images (zero-copy fast path).
    uint8_t*    d_grid[2]      = {};
    // Display buffers actually bound to the Image views: reduced-res interop buffers when reducing,
    // otherwise aliased onto the interop grid buffers.
    uint8_t*    d_disp[2]      = {};
    AllocHandle disp_alloc[2]  = {};
    ViewHandle  view[2]        = {};
    int r = 0, w = 1;    // sim ping-pong
    int dr = 0, dw = 1;  // display ping-pong (kept in lockstep with r/w)

    if (input.display)
    {
        if (reduce)
        {
            checkCuda(cudaMalloc(&d_grid[0], N));
            checkCuda(cudaMalloc(&d_grid[1], N));
            for (int k = 0; k < 2; ++k)
                allocLinear(instance, (void**)&d_disp[k], DN, &disp_alloc[k]);
        }
        else
        {
            for (int k = 0; k < 2; ++k)
                allocLinear(instance, (void**)&d_grid[k], N, &disp_alloc[k]);
            d_disp[0] = d_grid[0];
            d_disp[1] = d_grid[1];
        }

        // Generate the initial state directly on the GPU (no multi-GB host RNG fill / H2D copy).
        launchInitGrid(d_grid[0], W, H, input.ca.seed, input.ca.density);
        checkCuda(cudaDeviceSynchronize());

        // Seed the first display buffer so frame 0 already shows the initial state (whole grid).
        if (reduce)
        {
            launchResample(d_grid[0], d_disp[0], W, H, DW, DH, 0, 0, W, H);
            checkCuda(cudaDeviceSynchronize());
        }

        // Views are bound to the display buffers at display resolution (DW x DH == W x H when
        // not reducing), so this path is identical to the original when the grid fits.
        for (int k = 0; k < 2; ++k)
        {
            ViewDescription desc{
                .type       = ViewType::Image,
                .options    = {},
                .domain     = DomainType::Domain2D,
                .attributes = {
                    { AttributeType::Position, makeImageFrame(instance) },
                    { AttributeType::Color, AttributeDescription{
                        .source = disp_alloc[k],
                        .size   = (unsigned int)DN,
                        .format = r8_fmt,
                    }}
                },
                .layout  = Layout::make(DW, DH),
                .visible = (k == dr),  // only view[dr] starts visible
            };
            createView(instance, &desc, &view[k]);
        }
    }
    else
    {
        checkCuda(cudaMalloc(&d_grid[0], N));
        checkCuda(cudaMalloc(&d_grid[1], N));
        launchInitGrid(d_grid[0], W, H, input.ca.seed, input.ca.density);
    }

    // Ctrl+W closes the window; set by the GUI callback, polled by the main loop.

    // Clipmap view state for panning/zooming a grid too large to present at native resolution.
    // Written by the scroll callback (zoom) and the loop's key polling (pan), read by the loop to
    // place the sample window. (ox,oy) = top-left cell; vw = visible width in cells (vh derived to match the display
    // aspect). Smaller vw = deeper zoom (down to a few cells, magnified); vw == W shows the grid.
    struct ClipView { std::atomic<int> ox{0}, oy{0}, vw{1}; };
    ClipView clip;
    clip.vw.store(W);
    const int VW_MIN = std::min(W, 8);  // deepest zoom: as few as 8 cells across the display

    // HUD data collected each frame and pushed to the built-in overlay via setHudText.
    HudData hud{};
    hud.cells     = (unsigned int)N;
    hud.grid_w    = W;
    hud.grid_h    = H;
    hud.density   = input.ca.density;
    hud.seed      = input.ca.seed;
    hud.reduce    = reduce;
    hud.disp_w    = DW;
    hud.disp_h    = DH;
    hud.view_ox   = 0;
    hud.view_oy   = 0;
    hud.view_vw   = W;
    hud.view_vh   = H;

    if (input.display)
    {
        // GPU name/total-VRAM need NVML (filled after GPUPowerBegin below). The VRAM breakdown is
        // ready now: external + cuda_ctx are measured (checkpoints), buf + render + vulkan computed.
        hud.vram_external_mb = (float)vram_external;
        hud.cuda_ctx_mb      = (float)cuda_ctx_mb;
        // 2 x full-res uint8 sim grid, plus (when reducing) 2 x reduced-res uint8 display buffers.
        hud.buf_mb    = (float)((2 * N + (reduce ? 2 * DN : 0)) / (1024.0 * 1024.0));
        hud.render_mb = 0.f;  // image view samples the interop grid directly; no extra geometry
        // Vulkan render targets mimir's swapchain allocates, no MSAA: 3 swapchain (B8G8R8A8, 4 B)
        // + 1 depth (D32_SFLOAT, 4 B), each win_width*win_height.
        const double px = (double)input.win_width * (double)input.win_height;
        hud.vulkan_mb = (float)(px * 4.0 * (3 + 1) / (1024.0 * 1024.0));
        int device_id = -1;
        checkCuda(cudaGetDevice(&device_id));
        cudaDeviceProp prop{};
        checkCuda(cudaGetDeviceProperties(&prop, device_id));
        snprintf(hud.gpu_device, sizeof(hud.gpu_device), "%d (CC %d.%d)",
            device_id, prop.major, prop.minor);

        // Mouse-wheel zoom of the clipmap window. No ImGui: setScrollCallback delivers the wheel
        // delta directly (pan/reset are polled in the loop below via isKeyDown/isKeyPressed).
        if (reduce)
        {
            setScrollCallback(instance, [&clip, W, H, DW, DH, VW_MIN](double, double dy) {
                if (dy == 0.0) { return; }
                auto vh_of = [&](int vw) { int v = (int)((long)vw * DH / DW); return v < 1 ? 1 : v; };
                int cvw = clip.vw.load();
                int nvw = dy > 0.0 ? cvw - std::max(1, cvw / 8)   // zoom in (shrink window)
                                   : cvw + std::max(1, cvw / 8);  // zoom out (grow window)
                nvw = std::clamp(nvw, VW_MIN, W);
                if (nvw == cvw) { return; }
                int cvh = vh_of(cvw), nvh = vh_of(nvw);
                long cx = clip.ox.load() + (long)cvw / 2, cy = clip.oy.load() + (long)cvh / 2;
                clip.vw.store(nvw);
                clip.ox.store((int)std::clamp(cx - nvw / 2, 0L, (long)std::max(0, W - nvw)));
                clip.oy.store((int)std::clamp(cy - nvh / 2, 0L, (long)std::max(0, H - nvh)));
            });
        }
    }

    // CUDA timing events. pstart/pstop bracket the downsample ("pack") kernel when reducing.
    cudaEvent_t cstart = nullptr, cstop = nullptr, cstop_prev = nullptr;
    cudaEvent_t pstart = nullptr, pstop = nullptr;
    checkCuda(cudaEventCreate(&cstart));
    checkCuda(cudaEventCreate(&cstop));
    if (reduce && input.display)
    {
        checkCuda(cudaEventCreate(&pstart));
        checkCuda(cudaEventCreate(&pstop));
    }
    if (input.enable_interop_sync && input.display)
    {
        checkCuda(cudaEventCreate(&cstop_prev));
        checkCuda(cudaEventRecord(cstop_prev));
        checkCuda(cudaEventSynchronize(cstop_prev));
    }

    GPUPowerBegin("gpu", 100);
    // Headless never allocates display buffers, so report the plain 2xN grid there (ds = 1).
    printSystemInfo(input, DN, input.display && reduce, DW, DH);

    if (input.display)
    {
        // NVML is initialized by GPUPowerBegin() above, so the GPU name/VRAM queries succeed
        // here (they returned 0 when issued before init). Start the async render loop only
        // after the HUD fields are valid so the very first frame shows correct values.
        nvmlDeviceGetName(getNvmlDevice(), hud.gpu_name, sizeof(hud.gpu_name));
        nvmlMemory_v2_t mi;
        mi.version = (unsigned int)(sizeof(mi) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
        hud.gpu_total_gb = (float)(mi.total / (1024.0 * 1024.0 * 1024.0));

        displayAsync(instance);
    }

    using Clock = std::chrono::steady_clock;
    auto frame_start = Clock::now();
    auto loop_start  = Clock::now();
    size_t frame_count = 0;
    // Accumulate the benchmark's own per-frame kernel time (ms). The engine's compute_monitor only
    // ticks inside prepareViews/updateViews when enable_interop_sync is on, so in async mode
    // (--interop-sync 0) getMetrics().times.compute would be 0. Feed this measured total (seconds)
    // into the returned metrics below, matching benchmark_datoviz's compute semantics.
    double total_compute_ms = 0.0;
    double total_pack_ms    = 0.0;
    // Sum the per-frame render-pass GPU time into a TOTAL (seconds); getMetrics().times.pipeline is
    // the LAST frame's value, so summing it keeps pipeline_time consistent with compute/graphics.
    double total_pipeline_s = 0.0;
    int    iters_run        = 0;

    for (int i = 0; i < input.iter_count && (!input.display || isRunning(instance)); ++i) // Ctrl+W quits
    {
        iters_run = i + 1;
        // Clipmap pan/reset, polled from the input API (no ImGui): arrows/WASD pan, R resets to the
        // whole grid. Wheel-zoom is handled by the scroll callback above.
        if (input.display && reduce)
        {
            auto vh_of = [&](int vw) { int v = (int)((long)vw * DH / DW); return v < 1 ? 1 : v; };
            if (isKeyPressed(instance, Key::R))
            {
                clip.vw.store(W); clip.ox.store(0); clip.oy.store(0);
            }
            else
            {
                int vw = clip.vw.load(), vh = vh_of(vw);
                int stepx = std::max(1, vw / 20), stepy = std::max(1, vh / 20);
                int nox = clip.ox.load(), noy = clip.oy.load();
                if (isKeyDown(instance, Key::Right) || isKeyDown(instance, Key::D)) { nox += stepx; }
                if (isKeyDown(instance, Key::Left)  || isKeyDown(instance, Key::A)) { nox -= stepx; }
                if (isKeyDown(instance, Key::Down)  || isKeyDown(instance, Key::S)) { noy += stepy; }
                if (isKeyDown(instance, Key::Up)    || isKeyDown(instance, Key::W)) { noy -= stepy; }
                clip.ox.store(std::clamp(nox, 0, std::max(0, W - vw)));
                clip.oy.store(std::clamp(noy, 0, std::max(0, H - vh)));
            }
        }
        if (input.display) prepareViews(instance);

        if (input.display) checkCuda(cudaEventRecord(cstart));
        launchStepGoL(d_grid[r], d_grid[w], W, H);
        if (input.display) checkCuda(cudaEventRecord(cstop));

        // When the grid is larger than a single image can hold, downsample the just-written state
        // into the display buffer. Same stream as the step, so it is ordered after it. When not
        // reducing, d_disp aliases the grid buffers and no pack pass runs (zero-copy path).
        int vox = 0, voy = 0, vvw = W, vvh = H;  // current clipmap view, read from the GUI thread
        if (input.display && reduce)
        {
            vox = clip.ox.load(); voy = clip.oy.load();
            vvw = clip.vw.load(); vvh = std::max(1, (int)((long)vvw * DH / DW));
            checkCuda(cudaEventRecord(pstart));
            launchResample(d_grid[w], d_disp[dw], W, H, DW, DH, vox, voy, vvw, vvh);
            checkCuda(cudaEventRecord(pstop));
        }

        if (input.display)
        {
            // Switch visibility: hide the source display buffer, show the just-written one.
            toggleVisibility(view[dr]);
            toggleVisibility(view[dw]);
        }

        std::swap(r, w);
        if (input.display) std::swap(dr, dw);

        if (input.display)
        {
            updateViews(instance);

            // Sync on the last GPU event this frame (pack when reducing, else the step) so both
            // timings below are ready.
            checkCuda(cudaEventSynchronize(reduce ? pstop : cstop));
            float kernel_ms = 0.f;
            checkCuda(cudaEventElapsedTime(&kernel_ms, cstart, cstop));
            total_compute_ms += kernel_ms;

            float pack_ms = 0.f;
            if (reduce)
            {
                checkCuda(cudaEventElapsedTime(&pack_ms, pstart, pstop));
                total_pack_ms += pack_ms;
            }

            // Render time: from end of previous frame's compute region to start of this frame's
            // compute — i.e. the time Vulkan spent rendering frame (i-1). When reducing, this
            // interval also contains the downsample pass, so the on-screen Render figure is a
            // slight over-estimate; the CSV graphics_time (from the engine) is unaffected.
            if (input.enable_interop_sync && i > 0)
            {
                float render_ms = 0.f;
                checkCuda(cudaEventElapsedTime(&render_ms, cstop_prev, cstart));
                hud.render_ms = (i == 1) ? render_ms
                                         : 0.9f * hud.render_ms + 0.1f * render_ms;
            }
            if (input.enable_interop_sync) std::swap(cstop, cstop_prev);

            auto now = Clock::now();
            using ms = std::chrono::duration<float, std::milli>;
            float frame_ms = ms(now - frame_start).count();
            float new_fps  = frame_ms > 0.f ? 1000.f / frame_ms : 0.f;
            hud.frame      = i;
            hud.view_ox    = vox;  // reflect live zoom/pan in the HUD
            hud.view_oy    = voy;
            hud.view_vw    = vvw;
            hud.view_vh    = vvh;
            hud.pack_ms    = (i == 0) ? pack_ms : 0.9f * hud.pack_ms + 0.1f * pack_ms;
            hud.compute_ms = (i == 0) ? kernel_ms : 0.9f * hud.compute_ms + 0.1f * kernel_ms;
            hud.fps        = (i == 0) ? new_fps   : 0.9f * hud.fps        + 0.1f * new_fps;
            // Render sub-costs from the engine (GPU frame latency + CPU phases), same EMA
            // smoothing as the rest of the HUD.
            auto gt = getMetrics(instance).times;
            total_pipeline_s += gt.pipeline; // per-frame render-pass seconds -> accumulate to a total
            hud.wait_ms    = (i == 0) ? gt.wait   : 0.9f * hud.wait_ms   + 0.1f * gt.wait;
            hud.record_ms  = (i == 0) ? gt.record : 0.9f * hud.record_ms + 0.1f * gt.record;
            hud.submit_ms  = (i == 0) ? gt.submit : 0.9f * hud.submit_ms + 0.1f * gt.submit;
            hud.gpu_ms     = (i == 0) ? gt.gpu    : 0.9f * hud.gpu_ms    + 0.1f * gt.gpu;
            float watts    = (float)getGPUCurrentPower();
            hud.gpu_watts  = (i == 0) ? watts     : 0.9f * hud.gpu_watts  + 0.1f * watts;

            // Live VRAM-in-use (NVML), sampled every 30 frames to keep the query off the hot path.
            if (i == 0 || (i % 30) == 0)
                hud.vram_used_mb = (float)sampleVramUsedMB();
            frame_start    = now;
            ++frame_count;
            setHudText(instance, formatHud(hud).c_str()); // push metrics to the built-in overlay
        }
    }

    if (!input.display) checkCuda(cudaDeviceSynchronize());

    auto loop_wall = std::chrono::duration<double>(Clock::now() - loop_start).count();
    float frame_rate = 0.f;
    if (input.display && loop_wall > 0.0) frame_rate = (float)(frame_count / loop_wall);
    else if (loop_wall > 0.0)             frame_rate = (float)(input.iter_count / loop_wall);

    auto metrics = getMetrics(instance);

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

    // Cleanup
    if (cstart)     cudaEventDestroy(cstart);
    if (cstop)      cudaEventDestroy(cstop);
    if (cstop_prev) cudaEventDestroy(cstop_prev);
    if (pstart)     cudaEventDestroy(pstart);
    if (pstop)      cudaEventDestroy(pstop);

    if (input.display)
    {
        exit(instance);
        destroyInstance(instance);
        // Display buffers (disp_alloc) are interop allocations owned by mimir; freed via
        // destroyInstance. When reducing, the full-res sim grids are plain CUDA and are ours to free.
        if (reduce)
        {
            checkCuda(cudaFree(d_grid[0]));
            checkCuda(cudaFree(d_grid[1]));
        }
    }
    else
    {
        destroyInstance(instance);
        checkCuda(cudaFree(d_grid[0]));
        checkCuda(cudaFree(d_grid[1]));
    }

    metrics.frame_rate = frame_rate;
    // Override compute with the benchmark's measured total (seconds); the engine only populates
    // times.compute in interop-sync mode, so this makes async runs report correctly too.
    if (input.display) metrics.times.compute = (float)(total_compute_ms / 1000.0);
    // Pipeline: whole-run total render-pass GPU time in seconds (see total_pipeline_s), consistent
    // with compute/graphics. 0 in no-display mode (no render pass ran).
    metrics.times.pipeline = (float)total_pipeline_s;
    return BenchmarkResult{
        .perf = metrics, .power = gpu_power, .memory = nvml,
        .pack_time = (float)(total_pack_ms / 1000.0),
        .iters = iters_run,
    };
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
        "  --present N        0=Immediate 1=TripleBuffering 2=VSync (default: 0)\n"
        "                     Real display vsync lives here (--present 2).\n"
        "  --interop-sync N   CUDA-Vulkan interop sync: 1=on 0=off  (default: 1)\n"
        "                     NOT vsync; gates compute/render on the shared buffer.\n"
        "  --display N        1 = open window, 0 = headless compute (default: 1)\n"
        "\n"
        "Keys: F1 toggles the HUD (and every other GUI window) for clean screenshots;\n"
        "      Ctrl+G shows the engine scene-parameters panel; Ctrl+W/Ctrl+Q quit.\n"
        "\n"
        "Large grids: a grid larger than the device's max 2D image size cannot be presented as one\n"
        "      image, so it is shown through a downsampled clipmap window. Navigate it with the\n"
        "      mouse wheel (zoom), arrow keys (pan) and R (reset to whole grid).\n"
        "\n"
        "Frame rate is always uncapped (no target_fps limiter).\n"
        "Output: one CSV row to stdout.\n"
        "        Column layout matches benchmark_datoviz; d2h/h2h columns are 0. pack_time is the\n"
        "        grid->display downsample when the grid is too large to present natively, else 0.\n"
        "        graphics_time = mimir internal render time (zero-copy, no upload).\n",
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
            if      (a == "--present")      input.present = static_cast<PresentMode>(std::stoi(v));
            else if (a == "--interop-sync") input.enable_interop_sync = (bool)std::stoi(v);
            else if (a == "--display")      input.display = (bool)std::stoi(v);
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
