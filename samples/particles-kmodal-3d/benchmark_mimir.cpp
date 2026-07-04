// 3D random-walk point cloud benchmark — mimir rendering path.
//
// The position buffer is a mimir interop allocation: CUDA integrates the points in
// place and Vulkan reads the same buffer each frame — no pack step, no extra memory,
// zero per-frame host transfer.
//
// Compare with benchmark_datoviz.cpp which must round-trip GPU -> host -> GPU.

#include <atomic>
#include <chrono>
#include <cstring>
#include <vector>
#include <string>

#include <imgui.h>

#include "kmodal_sim.cuh"
#include "nvmlPower.hpp"
#include "validation.hpp"

#include <mimir/mimir.hpp>
using namespace mimir;

// ---------------------------------------------------------------------------
// Input / output structs
// ---------------------------------------------------------------------------

struct PointsInput {
    int          win_width   = 1920;
    int          win_height  = 1080;
    PointsParams pts         = {};          // point count, seed
    int          iter_count  = 1000000;
    PresentMode  present     = PresentMode::Immediate;
    bool         enable_interop_sync = true;
    bool         display     = true;
    float        size_px     = 5.f;         // marker size (pixels in unlit/None mode)
    LightModel   light_model = LightModel::None;  // None=Flat2D discs, Phong=lit spheres,
                                                  // PathTracing=RT (falls back to Phong for now)
    // Path-tracing knobs (only used with --light-model path-tracing).
    unsigned int pt_spp      = 1;            // samples per pixel per frame (--spp)
    unsigned int pt_bounces  = 4;            // max path depth (--bounces; effective in Phase 2)
    unsigned int pt_subdiv   = 1;            // icosphere tessellation 0/1/2 (--subdiv)
    float3       background  = { 0.f, 0.f, 0.f }; // window/environment color (--background)
    float3       pcolor      = { 0.82f, 0.82f, 0.88f }; // particle color (--pcolor)
    // Camera: interactive fly (WASD + captured mouse-look) vs the default orbit drag controls,
    // and a scripted auto-orbit for reproducible input-free runs.
    bool         fly         = false;        // --fly: FPS fly camera
    float        orbit_speed = 0.f;          // --orbit-speed DEG/S: scripted auto-orbit (0=off)
    float        cam_speed   = 3.f;          // --cam-speed U/S: WASD move speed
    float        sensitivity = 0.1f;         // --sensitivity DEG/PX: mouse-look sensitivity
};

// Parse --background as "G" (grey level) or "R,G,B" in [0,1].
static float3 parseColor(const std::string& v)
{
    float r = 0.f, g = 0.f, b = 0.f;
    if (std::sscanf(v.c_str(), "%f,%f,%f", &r, &g, &b) == 3) { return { r, g, b }; }
    float grey = std::stof(v);
    return { grey, grey, grey };
}

// Parse --light-model none|phong|path-tracing into the instance-wide LightModel.
static LightModel parseLightModel(const std::string& v)
{
    if (v == "none")         return LightModel::None;
    if (v == "phong")        return LightModel::Phong;
    if (v == "phong-mesh")   return LightModel::PhongMesh;
    if (v == "path-tracing") return LightModel::PathTracing;
    fprintf(stderr, "Unknown --light-model '%s' (use none|phong|path-tracing)\n", v.c_str());
    exit(EXIT_FAILURE);
}

struct HudData {
    unsigned int points;
    int          frame;
    float        fps;
    float        compute_ms;
    float        render_ms;
    float        pipeline_ms;    // GPU render-pass time (raster draw / PT composite) -- all modes
    float        tlas_ms;        // path tracing: per-frame TLAS rebuild (sub-metric of Render)
    float        trace_ms;       // path tracing: per-frame ray trace  (sub-metric of Render)
    float        wait_ms;        // CPU blocked on fence + swapchain acquire (GPU/present backpressure)
    float        record_ms;      // CPU command-buffer recording
    float        submit_ms;      // CPU vkQueueSubmit + vkQueuePresentKHR
    float        gpu_ms;         // true end-to-end GPU frame latency (submit -> fence signalled)
    bool         path_tracing;   // show the TLAS/Trace sub-metrics only under path tracing
    bool         fly;            // Fly camera active: show the TAB (release cursor) hint
    float        gpu_watts;
    char         gpu_name[256];
    char         gpu_device[64];   // "N (CC major.minor)"
    float        gpu_total_gb;
    float        buf_mb;
    uint32_t     seed;
    unsigned int k;
    float        epsilon;
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
    fprintf(stderr, "  positions  (interop):  %s\n", smb(pos_bytes).c_str());
    fprintf(stderr, "  rng states (CUDA):     %s\n", smb(rng_bytes).c_str());
    fprintf(stderr, "  clusters   (CUDA):     %s\n", smb(cluster_bytes).c_str());
    fprintf(stderr, "  Total:                 %s\n",
        smb(pos_bytes + rng_bytes + cluster_bytes).c_str());
}

void formatResults(PointsInput input, BenchmarkResult result)
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
        {"pack_time",     sf(0.f)},
        {"d2h_time",      sf(0.f)},
        {"h2h_time",      sf(0.f)},
        // Path-tracing columns (appended so the leading columns stay datoviz-comparable). For
        // non-path-tracing modes spp/bounces/subdiv carry their defaults and the times are 0.
        {"spp",           su(input.pt_spp)},
        {"bounces",       su(input.pt_bounces)},
        {"subdiv",        su(input.pt_subdiv)},
        {"tlas_time",     sf(lib.times.tlas_build)},
        {"trace_time",    sf(lib.times.trace)},
    });
    printf("%s,%s,%u,%u,%u,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%u,%u,%u,%f,%f\n",
        mode.c_str(), resolution.c_str(),
        input.pts.count, input.pts.seed, input.pts.k, input.pts.epsilon,
        lib.frame_rate, lib.times.compute, lib.times.pipeline, lib.times.graphics,
        lib.devmem.usage, lib.devmem.budget,
        gpu.average_power, gpu.total_energy, gpu.total_time,
        nvml.free, nvml.reserved, nvml.total, nvml.used,
        0.f, 0.f, 0.f,
        input.pt_spp, input.pt_bounces, input.pt_subdiv, lib.times.tlas_build, lib.times.trace);
}

// ---------------------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------------------

BenchmarkResult runExperiment(PointsInput input)
{
    const size_t n         = input.pts.count;
    const size_t pos_bytes = sizeof(float) * 3 * n;

    checkCuda(cudaSetDevice(0));

    // Mimir instance.
    ViewerOptions opts{};
    opts.window.size         = { input.win_width, input.win_height };
    opts.light_model         = input.light_model;
    // Sun coming from a diagonal behind-and-above the camera, so the home view is lit
    // like the datoviz baseline. The raster (Phong) path transforms light_pos into eye
    // space where +z faces the camera; the path tracer uses it as a raw WORLD direction,
    // where the camera-facing side has normal -z. So the two need opposite z signs to
    // light the same (visible) side. marker.slang uses dot(normal, light_pos) WITHOUT
    // normalizing, so it must stay unit length or it also scales diffuse brightness.
    opts.light_pos           = (input.light_model == LightModel::PathTracing)
        ? float3{ -0.4082f, 0.4082f, -0.8165f }  // world-space: from behind the camera
        : float3{ -0.4082f, 0.4082f,  0.8165f }; // eye-space (Phong/datoviz): normalize({-1,1,2})
    opts.background_color    = { input.background.x, input.background.y, input.background.z, 1.f };
    opts.pt_samples_per_pixel = input.pt_spp;
    opts.pt_max_bounces       = input.pt_bounces;
    opts.pt_subdivisions      = input.pt_subdiv;
    opts.present.mode              = input.present;
    opts.present.enable_interop_sync       = input.enable_interop_sync;
    opts.present.enable_fps_limit  = false;  // always uncapped
    opts.show_panel          = input.display;
    opts.camera_control      = input.fly ? CameraControl::Fly : CameraControl::Orbit;
    opts.mouse_sensitivity   = input.sensitivity;
    opts.camera_move_speed   = input.cam_speed;
    opts.orbit_speed         = input.orbit_speed;

    InstanceHandle instance = nullptr;
    createInstance(opts, &instance);

    float* d_pos = nullptr;
    if (input.display)
    {
        AllocHandle pos_alloc{};
        allocLinear(instance, (void**)&d_pos, pos_bytes, &pos_alloc);

        // marker_opts.render_mode is engine-managed now: createView derives it from
        // the instance's light_model (None -> Flat2D discs, Phong/PathTracing ->
        // Sphere3D). We must NOT set render_mode here.
        //
        // In None mode the markers are native point sprites sized in pixels — the
        // same semantics as datoviz markers, so --size means the same thing in both
        // benchmarks. Lit modes draw world-space spheres, where default_size is a
        // world-unit radius instead.
        MarkerOptions marker_opts = MarkerOptions::defaults();
        float size = (input.light_model == LightModel::None) ? input.size_px
                                                             : input.size_px / 100.f;

        ViewDescription desc{
            .type       = ViewType::Markers,
            .options    = marker_opts,
            .domain     = DomainType::Domain3D,
            .attributes = {
                { AttributeType::Position, AttributeDescription{
                    .source = pos_alloc,
                    .size   = (unsigned int)n,
                    .format = FormatDescription::make<float3>(),
                }}
            },
            .layout        = Layout::make((unsigned int)n),
            .default_color = { input.pcolor.x, input.pcolor.y, input.pcolor.z, 1.f },
            .default_size  = size,
            .linewidth     = 0.f,
            .scale         = { 1.f, 1.f, 1.f },
        };
        ViewHandle view = nullptr;
        createView(instance, &desc, &view);

        // Points live in [-1,1]^3; pull the camera back so the cube fits the frame.
        setCameraPosition(instance, { 0.f, 0.f, -4.f });
    }
    else
    {
        checkCuda(cudaMalloc((void**)&d_pos, pos_bytes));
    }

    auto rng      = createRngStates(input.pts.seed);
    auto clusters = createClusters(input.pts);
    launchInitPositions(d_pos, input.pts, clusters, rng);

    // Ctrl+W closes the window; set by the GUI callback, polled by the main loop.
    std::atomic<bool> quit_flag{false};

    // HUD data shared between the simulation loop and the ImGui callback.
    HudData hud{};
    hud.points  = (unsigned int)n;
    hud.seed    = input.pts.seed;
    hud.k       = input.pts.k;
    hud.epsilon = input.pts.epsilon;
    hud.path_tracing = (input.light_model == LightModel::PathTracing);
    hud.fly          = input.fly;

    if (input.display)
    {
        // GPU name/VRAM need NVML, which is only initialized later by GPUPowerBegin(); they are
        // filled in after that call. buf_mb and the CUDA device info need no NVML.
        hud.buf_mb = (float)((pos_bytes + rngStatesBytes(rng) + clusterBytes(clusters, n))
            / (1024.0 * 1024.0));
        int device_id = -1;
        checkCuda(cudaGetDevice(&device_id));
        cudaDeviceProp prop{};
        checkCuda(cudaGetDeviceProperties(&prop, device_id));
        snprintf(hud.gpu_device, sizeof(hud.gpu_device), "%d (CC %d.%d)",
            device_id, prop.major, prop.minor);

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
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Device");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%s", hud.gpu_device);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("VRAM");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f GB", hud.gpu_total_gb);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Points");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", hud.points);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Seed");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", hud.seed);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Modes k");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%u", hud.k);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Epsilon");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.4f", hud.epsilon);
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
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Transfer");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    Pack");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    D2H");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    H2H");
                ImGui::TableSetColumnIndex(1); ImGui::TextUnformatted("N/A");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Render");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.render_ms);
                // Render sub-costs (indented, like Transfer's Pack/D2H/H2H). Render is the render
                // thread's wall time per frame. "GPU frame" is the true end-to-end GPU latency
                // (submit -> fence signalled) and normally accounts for essentially all of Render
                // in lockstep interop mode. It is measured only on frames where CUDA had already
                // finished at submit time, so it is pure render work with no compute-wait folded in.
                // The CPU phases (Wait/Record/Submit) show the render thread does NOT block on the
                // CPU -- the cost is the GPU executing the draw, which the compute thread then waits
                // on through the interop handshake (that wait is what Render measures).
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    GPU frame");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.gpu_ms);
                if (hud.path_tracing) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    TLAS build");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.tlas_ms);
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    Trace");
                    ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.trace_ms);
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    Wait (cpu)");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.wait_ms);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    Record (cpu)");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.record_ms);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("    Submit (cpu)");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f ms", hud.submit_ms);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted("Power");
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.1f W", hud.gpu_watts);
                ImGui::EndTable();
            }
            if (hud.fly) {
                ImGui::Separator();
                ImGui::TextDisabled("TAB: release cursor for menus");
            }
            ImGui::End();
        });
    }

    // CUDA timing events.
    cudaEvent_t cstart = nullptr, cstop = nullptr, cstop_prev = nullptr;
    checkCuda(cudaEventCreate(&cstart));
    checkCuda(cudaEventCreate(&cstop));
    if (input.enable_interop_sync && input.display)
    {
        checkCuda(cudaEventCreate(&cstop_prev));
        checkCuda(cudaEventRecord(cstop_prev));
        checkCuda(cudaEventSynchronize(cstop_prev));
    }

    GPUPowerBegin("gpu", 100);
    printSystemInfo(input, rngStatesBytes(rng), clusterBytes(clusters, n));

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

    for (int i = 0; i < input.iter_count && (!input.display || (isRunning(instance) && !quit_flag)); ++i)
    {
        if (input.display) prepareViews(instance);

        if (input.display) checkCuda(cudaEventRecord(cstart));
        launchIntegrate3D(d_pos, n, clusters, rng);
        if (input.display) checkCuda(cudaEventRecord(cstop));

        if (input.display)
        {
            updateViews(instance);

            checkCuda(cudaEventSynchronize(cstop));
            float kernel_ms = 0.f;
            checkCuda(cudaEventElapsedTime(&kernel_ms, cstart, cstop));

            // Render time: from end of previous frame's compute to start of this
            // frame's compute — i.e. the time Vulkan spent rendering frame (i-1).
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
            hud.compute_ms = (i == 0) ? kernel_ms : 0.9f * hud.compute_ms + 0.1f * kernel_ms;
            hud.fps        = (i == 0) ? new_fps   : 0.9f * hud.fps        + 0.1f * new_fps;
            // GPU-timestamped render sub-costs (components within Render), same EMA smoothing.
            // pipeline is the render-pass GPU time for every light model; tlas/trace are the extra
            // ray-tracing passes under path tracing.
            auto gt = getMetrics(instance).times;
            hud.pipeline_ms = (i == 0) ? gt.pipeline : 0.9f * hud.pipeline_ms + 0.1f * gt.pipeline;
            hud.wait_ms     = (i == 0) ? gt.wait   : 0.9f * hud.wait_ms   + 0.1f * gt.wait;
            hud.record_ms   = (i == 0) ? gt.record : 0.9f * hud.record_ms + 0.1f * gt.record;
            hud.submit_ms   = (i == 0) ? gt.submit : 0.9f * hud.submit_ms + 0.1f * gt.submit;
            hud.gpu_ms      = (i == 0) ? gt.gpu    : 0.9f * hud.gpu_ms    + 0.1f * gt.gpu;
            if (hud.path_tracing)
            {
                hud.tlas_ms  = (i == 0) ? gt.tlas_build : 0.9f * hud.tlas_ms  + 0.1f * gt.tlas_build;
                hud.trace_ms = (i == 0) ? gt.trace      : 0.9f * hud.trace_ms + 0.1f * gt.trace;
            }
            float watts    = (float)getGPUCurrentPower();
            hud.gpu_watts  = (i == 0) ? watts     : 0.9f * hud.gpu_watts  + 0.1f * watts;
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
    destroyClusters(clusters);
    destroyRngStates(rng);

    if (input.display)
    {
        exit(instance);
        destroyInstance(instance);
        // d_pos is an interop allocation owned by mimir; freed via destroyInstance.
    }
    else
    {
        destroyInstance(instance);
        checkCuda(cudaFree(d_pos));
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
        "Usage: %s [win_w win_h] [points] [seed] [iters] [options]\n"
        "\n"
        "Positional (in order; win_w/win_h must be supplied together):\n"
        "  win_w  win_h   Window resolution in pixels             (default: 1920 1080)\n"
        "  points         Number of simulated points              (default: 1000000)\n"
        "  seed           RNG seed for positions/walk             (default: 12345)\n"
        "  iters          Simulation steps to run                 (default: 1000000)\n"
        "\n"
        "Options (named, order-independent; omitted ones use their default):\n"
        "  --present N        0=Immediate 1=TripleBuffering 2=VSync (default: 0)\n"
        "                     Real display vsync lives here (--present 2).\n"
        "  --interop-sync N   CUDA-Vulkan interop sync: 1=on 0=off  (default: 1)\n"
        "                     NOT vsync; gates compute/render on the shared buffer.\n"
        "  --display N        1 = open window, 0 = headless compute (default: 1)\n"
        "  --size S           Marker size in pixels (none) or /100 world radius\n"
        "                     (phong/path-tracing)                  (default: 5)\n"
        "                     In 'none' mode, same meaning as benchmark_datoviz --size.\n"
        "  --light-model M    none         = unlit Flat2D discs (datoviz-comparable),\n"
        "                     phong        = lit Sphere3D impostors (datoviz-comparable),\n"
        "                     phong-mesh   = lit instanced icosphere meshes (--subdiv, mimir-only;\n"
        "                                    cheaper than impostors, matches path-tracing geometry),\n"
        "                     path-tracing = Vulkan RT (falls back to phong for now)\n"
        "                                                            (default: none)\n"
        "  --k N              Gaussian modes (clusters) at init     (default: 8)\n"
        "  --epsilon E        Per-axis stddev of each mode          (default: 0.05)\n"
        "                     The walk is mean-reverting, so clusters keep this\n"
        "                     stddev over time. Centers are seed-deterministic;\n"
        "                     same cloud as benchmark_datoviz for equal args.\n"
        "  --background C     Window/environment color: grey 'G' or 'R,G,B' in [0,1]\n"
        "                     (default: 0 = black). Under path-tracing this is also the\n"
        "                     sky/miss and ambient fill; black = sun-only lighting.\n"
        "  --pcolor C         Particle color: grey 'G' or 'R,G,B' in [0,1] (default: light grey)\n"
        "  Camera (on-screen):\n"
        "  --fly              FPS fly camera: captured mouse-look + WASD (Q/E or Space/LCtrl\n"
        "                     for down/up); TAB releases the cursor for the HUD. Default is\n"
        "                     orbit drag (left=rotate, right=zoom, middle=pan).\n"
        "  --cam-speed U/S    WASD move speed in world units/second     (default: 3)\n"
        "  --sensitivity D/PX Mouse-look degrees per pixel              (default: 0.1)\n"
        "  --orbit-speed D/S  Scripted auto-orbit around the scene at D deg/s for input-free\n"
        "                     reproducible runs; overrides manual control (default: 0 = off)\n"
        "  Path-tracing only (--light-model path-tracing):\n"
        "  --spp N            Samples per pixel per frame (antialiasing) (default: 1)\n"
        "  --bounces N        Max path depth                          (default: 4)\n"
        "                     (indirect bounces take effect in Phase 2)\n"
        "  --subdiv N         Icosphere tessellation 0=20 1=80 2=320 tris (default: 1)\n"
        "\n"
        "Frame rate is always uncapped (no target_fps limiter).\n"
        "Output: one CSV row to stdout.\n"
        "        Leading columns match benchmark_datoviz; pack/d2h/h2h columns are 0.\n"
        "        graphics_time = mimir internal render time (zero-copy, no upload).\n"
        "        Appended path-tracing columns: spp,bounces,subdiv,tlas_time,trace_time\n"
        "        (tlas_time/trace_time are GPU ms, 0 for non-path-tracing modes).\n"
        "        See sweep_pathtracing.sh to sweep these knobs into a CSV.\n",
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
        if (a == "--fly") { input.fly = true; continue; } // valueless flag
        if (a.rfind("--", 0) == 0)
        {
            if (i + 1 >= argc)
            { fprintf(stderr, "Missing value for %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
            std::string v = argv[++i];
            if      (a == "--present")      input.present = static_cast<PresentMode>(std::stoi(v));
            else if (a == "--interop-sync") input.enable_interop_sync = (bool)std::stoi(v);
            else if (a == "--display")      input.display = (bool)std::stoi(v);
            else if (a == "--size")         input.size_px = std::stof(v);
            else if (a == "--light-model")  input.light_model = parseLightModel(v);
            else if (a == "--k")            input.pts.k = (unsigned int)std::stoul(v);
            else if (a == "--epsilon")      input.pts.epsilon = std::stof(v);
            else if (a == "--spp")          input.pt_spp = (unsigned int)std::stoul(v);
            else if (a == "--bounces")      input.pt_bounces = (unsigned int)std::stoul(v);
            else if (a == "--subdiv")       input.pt_subdiv = (unsigned int)std::stoul(v);
            else if (a == "--background")   input.background = parseColor(v);
            else if (a == "--pcolor")       input.pcolor = parseColor(v);
            else if (a == "--orbit-speed")  input.orbit_speed = std::stof(v);
            else if (a == "--cam-speed")    input.cam_speed = std::stof(v);
            else if (a == "--sensitivity")  input.sensitivity = std::stof(v);
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { pos.push_back(a); }
    }
    if (pos.size() >= 2) { input.win_width = std::stoi(pos[0]); input.win_height = std::stoi(pos[1]); }
    if (pos.size() >= 3)   input.pts.count = (unsigned int)std::stoul(pos[2]);
    if (pos.size() >= 4)   input.pts.seed  = (uint32_t)std::stoul(pos[3]);
    if (pos.size() >= 5)   input.iter_count = std::stoi(pos[4]);

    // Mesh spheres default to a smoother tessellation than path tracing's default; --subdiv still
    // overrides (pt_subdiv==1 is the struct default, i.e. not explicitly set on the CLI).
    if (input.light_model == LightModel::PhongMesh && input.pt_subdiv == 1) { input.pt_subdiv = 2; }

    auto result = runExperiment(input);
    formatResults(input, result);
    return EXIT_SUCCESS;
}
