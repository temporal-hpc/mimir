// 2D Game of Life benchmark — mimir rendering path.
//
// Both ping-pong uint8 grid buffers are allocated as mimir interop allocations.
// Vulkan reads the just-written buffer directly each frame — no pack step, no
// extra memory, zero per-frame host transfer.
//
// Compare with benchmark_datoviz.cpp which must round-trip GPU -> host -> GPU.

#include <atomic>
#include <chrono>
#include <vector>
#include <string>

#include <imgui.h>

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
    int         target_fps   = 0;
    bool        enable_sync  = true;
    bool        display      = true;
};

struct HudData {
    unsigned int cells;
    int          frame;
    float        fps;
    float        compute_ms;
    float        render_ms;
    char         gpu_name[256];
    float        gpu_total_gb;
    float        buf_mb;
    int          grid_w, grid_h;
    float        density;
    uint32_t     seed;
};

struct GPUMemoryMetrics { double free, reserved, total, used; };

struct BenchmarkResult {
    PerformanceMetrics perf;
    GPUPowerMetrics    power;
    GPUMemoryMetrics   memory;
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
    fprintf(stderr, "  d_grid[0]  (interop):  %s\n", smb(N).c_str());
    fprintf(stderr, "  d_grid[1]  (interop):  %s\n", smb(N).c_str());
    fprintf(stderr, "  Total:                 %s\n", smb(2 * N).c_str());
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

    // pack_time/d2h_time/h2h_time are 0 for mimir (zero-copy, no pack step).
    // graphics_time = mimir's internal render time.
    // Column layout matches benchmark_datoviz for direct CSV comparison.
    printAligned({
        {"mode",          mode},
        {"windowres",     resolution},
        {"grid_w",        sd(input.ca.width)},
        {"grid_h",        sd(input.ca.height)},
        {"seed",          su(input.ca.seed)},
        {"density",       sf(input.ca.density)},
        {"target_fps",    sd(input.target_fps)},
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
        {"pack_time",     sf(0.f)},
        {"d2h_time",      sf(0.f)},
        {"h2h_time",      sf(0.f)},
    });
    printf("%s,%s,%d,%d,%u,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(), resolution.c_str(),
        input.ca.width, input.ca.height, input.ca.seed, input.ca.density,
        input.target_fps,
        lib.frame_rate, lib.times.compute, lib.times.pipeline, lib.times.graphics,
        lib.devmem.usage, lib.devmem.budget,
        gpu.average_power, gpu.total_energy, gpu.total_time,
        nvml.free, nvml.reserved, nvml.total, nvml.used,
        0.f, 0.f, 0.f);
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

    // Mimir instance.
    ViewerOptions opts{};
    opts.window.size         = { input.win_width, input.win_height };
    opts.background_color    = { 0.f, 0.f, 0.f, 1.f };
    opts.present.mode        = input.present;
    opts.present.enable_sync = input.enable_sync;
    opts.present.target_fps  = input.target_fps;
    opts.show_panel          = input.display;

    InstanceHandle instance = nullptr;
    createInstance(opts, &instance);

    // Two ping-pong CA grid buffers (uint8).
    // In display mode both are interop allocations so Vulkan can read them directly.
    // In headless mode plain CUDA allocations suffice.
    uint8_t*    d_grid[2]      = {};
    AllocHandle grid_alloc[2]  = {};
    ViewHandle  view[2]        = {};
    int r = 0, w = 1;

    auto h_grid = initGrid(input.ca);

    if (input.display)
    {
        for (int k = 0; k < 2; ++k)
            allocLinear(instance, (void**)&d_grid[k], N, &grid_alloc[k]);

        // Load initial state into buffer 0.
        checkCuda(cudaMemcpy(d_grid[0], h_grid.data(), N, cudaMemcpyHostToDevice));
        checkCuda(cudaDeviceSynchronize());

        // R8 UNORM format: cells store 0 (dead) or 255 (alive).
        FormatDescription r8_fmt{ .kind = FormatKind::UnsignedNormalized, .size = 1, .components = 1 };

        for (int k = 0; k < 2; ++k)
        {
            ViewDescription desc{
                .type       = ViewType::Image,
                .options    = {},
                .domain     = DomainType::Domain2D,
                .attributes = {
                    { AttributeType::Position, makeImageFrame(instance) },
                    { AttributeType::Color, AttributeDescription{
                        .source = grid_alloc[k],
                        .size   = (unsigned int)N,
                        .format = r8_fmt,
                    }}
                },
                .layout  = Layout::make(W, H),
                .visible = (k == r),  // only view[r] starts visible
            };
            createView(instance, &desc, &view[k]);
        }
    }
    else
    {
        checkCuda(cudaMalloc(&d_grid[0], N));
        checkCuda(cudaMalloc(&d_grid[1], N));
        checkCuda(cudaMemcpy(d_grid[0], h_grid.data(), N, cudaMemcpyHostToDevice));
    }

    // Ctrl+W closes the window; set by the GUI callback, polled by the main loop.
    std::atomic<bool> quit_flag{false};

    // HUD data shared between the simulation loop and the ImGui callback.
    HudData hud{};
    hud.cells   = (unsigned int)N;
    hud.grid_w  = W;
    hud.grid_h  = H;
    hud.density = input.ca.density;
    hud.seed    = input.ca.seed;

    if (input.display)
    {
        nvmlDeviceGetName(getNvmlDevice(), hud.gpu_name, sizeof(hud.gpu_name));
        nvmlMemory_v2_t mi;
        mi.version = (unsigned int)(sizeof(mi) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &mi);
        hud.gpu_total_gb = (float)(mi.total / (1024.0 * 1024.0 * 1024.0));
        hud.buf_mb = (float)((2 * N) / (1024.0 * 1024.0));  // 2 × uint8 interop grid

        setGuiCallback(instance, [&hud, &quit_flag]() {
            auto& io = ImGui::GetIO();
            if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_W, /*repeat=*/false))
                quit_flag.store(true);

            ImVec2 disp = io.DisplaySize;
            ImGui::SetNextWindowPos(ImVec2(disp.x - 10.f, 10.f),
                ImGuiCond_Always, ImVec2(1.f, 0.f));
            ImGui::Begin("Performance", nullptr,
                ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove |
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
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Grid");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%d x %d", hud.grid_w, hud.grid_h);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Seed");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", hud.seed);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Density");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f", hud.density);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Buffers");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f MB", hud.buf_mb);
                ImGui::EndTable();
            }
            ImGui::Separator();
            if (ImGui::BeginTable("perf", 2)) {
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

        displayAsync(instance);
    }

    // CUDA timing events.
    cudaEvent_t cstart = nullptr, cstop = nullptr, cstop_prev = nullptr;
    checkCuda(cudaEventCreate(&cstart));
    checkCuda(cudaEventCreate(&cstop));
    if (input.enable_sync && input.display)
    {
        checkCuda(cudaEventCreate(&cstop_prev));
        checkCuda(cudaEventRecord(cstop_prev));
        checkCuda(cudaEventSynchronize(cstop_prev));
    }

    GPUPowerBegin("gpu", 100);
    printSystemInfo(input);

    using Clock = std::chrono::steady_clock;
    auto frame_start = Clock::now();
    auto loop_start  = Clock::now();
    size_t frame_count = 0;

    for (int i = 0; i < input.iter_count && (!input.display || (isRunning(instance) && !quit_flag)); ++i)
    {
        if (input.display) prepareViews(instance);

        if (input.display) checkCuda(cudaEventRecord(cstart));
        launchStepGoL(d_grid[r], d_grid[w], W, H);
        if (input.display) checkCuda(cudaEventRecord(cstop));

        if (input.display)
        {
            // Switch visibility: hide the source buffer, show the just-written one.
            toggleVisibility(view[r]);
            toggleVisibility(view[w]);
        }

        std::swap(r, w);

        if (input.display)
        {
            updateViews(instance);

            checkCuda(cudaEventSynchronize(cstop));
            float kernel_ms = 0.f;
            checkCuda(cudaEventElapsedTime(&kernel_ms, cstart, cstop));

            // Render time: from end of previous frame's pack kernel to start of this
            // frame's compute — i.e. the time Vulkan spent rendering frame (i-1).
            if (input.enable_sync && i > 0)
            {
                float render_ms = 0.f;
                checkCuda(cudaEventElapsedTime(&render_ms, cstop_prev, cstart));
                hud.render_ms = (i == 1) ? render_ms
                                         : 0.9f * hud.render_ms + 0.1f * render_ms;
            }
            if (input.enable_sync) std::swap(cstop, cstop_prev);

            auto now = Clock::now();
            using ms = std::chrono::duration<float, std::milli>;
            float frame_ms = ms(now - frame_start).count();
            float new_fps  = frame_ms > 0.f ? 1000.f / frame_ms : 0.f;
            hud.frame      = i;
            hud.compute_ms = (i == 0) ? kernel_ms : 0.9f * hud.compute_ms + 0.1f * kernel_ms;
            hud.fps        = (i == 0) ? new_fps   : 0.9f * hud.fps        + 0.1f * new_fps;
            frame_start    = now;
            ++frame_count;
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

    if (input.display)
    {
        exit(instance);
        destroyInstance(instance);
        // grid_alloc[0/1] are interop allocations owned by mimir; freed via destroyInstance.
    }
    else
    {
        destroyInstance(instance);
        checkCuda(cudaFree(d_grid[0]));
        checkCuda(cudaFree(d_grid[1]));
    }

    metrics.frame_rate = frame_rate;
    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml };
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void usage(const char* prog)
{
    printf(
        "Usage: %s [win_w win_h] [grid_w grid_h] [seed] [density] [iters]"
        " [present] [target_fps] [vsync] [display]\n"
        "\n"
        "  win_w  win_h   Window resolution in pixels             (default: 1920 1080)\n"
        "  grid_w grid_h  CA grid dimensions in cells             (default: 1024 1024)\n"
        "  seed           RNG seed for initial state              (default: 12345)\n"
        "  density        Initial live-cell fraction [0,1]        (default: 0.30)\n"
        "  iters          Simulation steps to run                 (default: 1000000)\n"
        "  present        0=Immediate 1=TripleBuffering 2=VSync   (default: 0)\n"
        "  target_fps     Frame-rate cap; 0 = unlimited           (default: 0)\n"
        "  vsync          1 = enable GPU sync, 0 = disable        (default: 1)\n"
        "  display        1 = open window, 0 = headless compute   (default: 1)\n"
        "\n"
        "win_w/win_h and grid_w/grid_h must each be supplied as a pair.\n"
        "Output: one CSV row to stdout.\n"
        "        Column layout matches benchmark_datoviz; pack/d2h/h2h columns are 0.\n"
        "        graphics_time = mimir internal render time (zero-copy, no upload).\n",
        prog);
}

int main(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")
            { usage(argv[0]); return EXIT_SUCCESS; }
    if (argc == 1) { usage(argv[0]); return EXIT_SUCCESS; }

    CAInput input{};
    if (argc >= 3)  { input.win_width   = std::stoi(argv[1]); input.win_height  = std::stoi(argv[2]); }
    if (argc >= 5)  { input.ca.width    = std::stoi(argv[3]); input.ca.height   = std::stoi(argv[4]); }
    if (argc >= 6)    input.ca.seed     = (uint32_t)std::stoul(argv[5]);
    if (argc >= 7)    input.ca.density  = std::stof(argv[6]);
    if (argc >= 8)    input.iter_count  = std::stoi(argv[7]);
    if (argc >= 9)    input.present     = static_cast<PresentMode>(std::stoi(argv[8]));
    if (argc >= 10)   input.target_fps  = std::stoi(argv[9]);
    if (argc >= 11)   input.enable_sync = (bool)std::stoi(argv[10]);
    if (argc >= 12)   input.display     = (bool)std::stoi(argv[11]);

    auto result = runExperiment(input);
    formatResults(input, result);
    return EXIT_SUCCESS;
}
