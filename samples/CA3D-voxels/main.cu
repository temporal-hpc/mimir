#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#define CA_LOW 2
#define CA_HIGH 3
#define CA_NACER 3

#include <iostream>
#include <string>    // std::string, std::stoul (--lod parsing)
#include <vector>    // std::vector (CA_LOD_CHECK host buffers)
#include <algorithm> // std::max (CA_LOD_CHECK reference)
#include <fstream>   // /proc/cpuinfo (system info)
#include <thread>    // std::thread::hardware_concurrency (system info)
#include <atomic>    // std::atomic (quit/pause flags shared with the GUI callback)
#include <sys/sysinfo.h> // sysinfo (total RAM)

#include "tools.h"
#include "kernel3D.cuh"
#include "openmp3D.h"

#include <mimir/mimir.hpp>
#include <imgui.h> // Performance HUD (setGuiCallback)
#include "validation.hpp" // checkCuda
using namespace mimir;

// Reference max-pool exported by libmimir (mimir/voxel_lod.hpp is a private header, so declare it here).
// Identical formula to the in-shader voxel_lod.slang pool; used only by the CA_LOD_CHECK gate below.
namespace mimir {
void voxelPoolMax(const int* fine, uint32_t N, int* coarse, uint32_t M, cudaStream_t stream);
}

static void print_usage(const char* prog)
{
    printf("Usage: %s n nt B seed steps prob modo [--lod M] [--fly]\n", prog);
    printf("           [--light-model MODE] [--opacity A]\n\n");
    printf("3D cellular automaton rendered as a voxel volume with mimir.\n\n");
    printf("Positional arguments:\n");
    printf("  n      grid size (N^3 cells)\n");
    printf("  nt     OpenMP threads (data init / CPU solver)\n");
    printf("  B      CUDA block size, BxBxB (B <= 10)\n");
    printf("  seed   RNG seed for the initial state\n");
    printf("  steps  number of simulation steps\n");
    printf("  prob   initial alive probability in [0,1]\n");
    printf("  modo   0 = CPU, 1 = GPU\n\n");
    printf("Options:\n");
    printf("  --lod M            coarsen the N^3 grid to M^3 in the vertex shader (M < N, GPU render)\n");
    printf("  --fly              FPS fly camera (WASD/QE + mouse-look, TAB releases cursor); default orbit\n");
    printf("  --light-model MODE  how living voxels are shaded (default flat):\n");
    printf("                       flat          opaque, unlit cubes\n");
    printf("                       phong         cubes lit by the scene light (better 3D shape)\n");
    printf("                       path-tracing  ray-traced boxes (needs an RT-capable GPU)\n");
    printf("  --opacity A        living-cell alpha in [0,1] (default 1). A < 1 makes the volume\n");
    printf("                     see-through (dead cells hidden, depth-write off) so interior\n");
    printf("                     living cells are visible. Applies to the chosen light mode.\n");
    printf("  --background-color C  window background: grey 'G' or 'R,G,B' in [0,1]\n");
    printf("  --cell-color C        living-cell color: grey 'G' or 'R,G,B' in [0,1]\n");
    printf("  -h, --help         show this help and exit\n\n");
    printf("Environment:\n");
    printf("  CA_SHOT=<path>     render one offscreen frame (initial state) to a PPM and exit (no window)\n");
    printf("  CA_LOD_CHECK=1     one-shot: assert the GPU max-pool matches a CPU reference (needs --lod)\n");
}

// Living-voxel shading selected by --light-model. Opacity is a separate --opacity knob.
enum class VoxLight { Flat, Phong, PathTracing };

// Print basic host/device info (GPU, CPU, RAM) at startup.
static void print_system_info()
{
    printf("System:\n");
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess)
    {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess)
            printf("  GPU : %s (CC %d.%d, %d SMs, %.1f GB VRAM)\n", p.name, p.major, p.minor,
                p.multiProcessorCount, (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    std::string cpu_model = "unknown";
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("model name", 0) == 0)
            {
                auto pos = line.find(':');
                if (pos != std::string::npos) cpu_model = line.substr(pos + 2);
                break;
            }
    }
    printf("  CPU : %s (%u hw threads)\n", cpu_model.c_str(), std::thread::hardware_concurrency());
    struct sysinfo si{};
    if (sysinfo(&si) == 0)
        printf("  RAM : %.1f GB\n",
            (double)si.totalram * (double)si.mem_unit / (1024.0 * 1024.0 * 1024.0));
}

// Parse a color as a single grey level "G" or "R,G,B", components in [0,1]. Returns false if neither.
static bool parse_color(const std::string& s, float3& out)
{
    float r, g, b;
    if (std::sscanf(s.c_str(), "%f,%f,%f", &r, &g, &b) == 3) { out = { r, g, b }; return true; }
    if (std::sscanf(s.c_str(), "%f", &r) == 1)               { out = { r, r, r }; return true; }
    return false;
}

int main(int argc, char **argv){
    // --help works without the positional args.
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return EXIT_SUCCESS; }
    }
    if(argc < 8){
        fprintf(stderr, "missing arguments\n\n");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    const char *map[2] = {"CPU", "GPU"};
    // args
    long n      = atoi(argv[1]);
    int nt      = atoi(argv[2]);
    int B       = atoi(argv[3]);
    int seed    = atoi(argv[4]);
    int steps   = atoi(argv[5]);
    float prob  = atof(argv[6]);
    int modo = atoi(argv[7]);
    // Optional flags after the 7 positional args (see print_usage).
    unsigned int lod_cells = 0;
    bool     fly     = false;
    VoxLight light   = VoxLight::Flat;
    float    opacity = 1.0f;
    float3   bg_color   = { 0.10f, 0.10f, 0.12f }; // window background (dark makes the volume pop)
    float3   cell_color = { 0.15f, 0.45f, 1.00f }; // living-cell color
    for (int i = 8; i < argc; ++i)
    {
        std::string a = argv[i];
        if      (a == "--lod" && i + 1 < argc) lod_cells = (unsigned)std::stoul(argv[++i]);
        else if (a == "--fly")                 fly = true;
        else if (a == "--opacity" && i + 1 < argc) opacity = std::stof(argv[++i]);
        else if (a == "--light-model" && i + 1 < argc)
        {
            std::string m = argv[++i];
            if      (m == "flat" || m == "none") light = VoxLight::Flat;
            else if (m == "phong")               light = VoxLight::Phong;
            else if (m == "path-tracing" || m == "pt") light = VoxLight::PathTracing;
            else { fprintf(stderr, "Unknown --light-model '%s' (flat|phong|path-tracing)\n", m.c_str());
                   return EXIT_FAILURE; }
        }
        else if (a == "--background-color" && i + 1 < argc)
        {
            if (!parse_color(argv[++i], bg_color)) {
                fprintf(stderr, "Bad --background-color (use 'G' or 'R,G,B' in [0,1])\n");
                return EXIT_FAILURE; }
        }
        else if (a == "--cell-color" && i + 1 < argc)
        {
            if (!parse_color(argv[++i], cell_color)) {
                fprintf(stderr, "Bad --cell-color (use 'G' or 'R,G,B' in [0,1])\n");
                return EXIT_FAILURE; }
        }
        else { fprintf(stderr, "Unknown argument '%s'\n", a.c_str()); print_usage(argv[0]);
               return EXIT_FAILURE; }
    }
    if (opacity < 0.f) opacity = 0.f;
    if (opacity > 1.f) opacity = 1.f;
    double t1;

    if(B > 10 || modo > 1){
        fprintf(stderr, "B must be <= 10 and modo must be 0 (CPU) or 1 (GPU)\n\n");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    print_system_info();

    // SETEO DE OPENMP THREADS (solo relevante para inicializar datos y solucion CPU)

    omp_set_num_threads(nt);
    // TODO CAMBIAR A 2D
    printf("modo: %s     n=%ld (%.3f GiBytes / cubo)    nt=%i   B=%i  steps=%i\n", map[modo], n, sizeof(int)*n*n*n/(1024*1024*1024.0), nt, B, steps);

    // original (3D)
    // TODO CAMBIAR A 2D
    int *original = new int[n*n*n];

    // punteros GPU (3D)
    int *d1, *d2;

    // CREACION DE DATOS
    printf("Inicializando.................."); fflush(stdout);
    t1 = omp_get_wtime();
    init_prob(n, original, seed, prob);

    int width = 1920, height = 1080;
    InstanceHandle instance = nullptr;
    ViewerOptions opts;
    opts.window.size  = {width, height};
    opts.pt_lod_cells = lod_cells; // 0 = no LOD; M > 0 coarsens a Voxels grid to M^3
    // Camera: orbit (left-drag rotate, right-drag zoom, middle-drag pan) by default, or a fly camera
    // with --fly. The fly move speed defaults to 3 world units/s, which is glacial across an N-unit
    // grid, so scale it to the grid size (TAB releases the captured cursor for the HUD).
    opts.camera_control = fly ? CameraControl::Fly : CameraControl::Orbit;
    if (fly) { opts.camera_move_speed = 0.6f * n; }
    // Living-voxel shading. flat = unlit, phong = lit cubes, path-tracing = ray-traced boxes (RT GPU).
    opts.light_model = (light == VoxLight::Phong)       ? LightModel::Phong
                     : (light == VoxLight::PathTracing) ? LightModel::PathTracing
                                                        : LightModel::None;
    opts.background_color = { bg_color.x, bg_color.y, bg_color.z, 1.f };
    // CA_SHOT=<path>: render one offscreen frame of the initial state to a PPM and exit (no window).
    // Handy for eyeballing the LOD framing / capturing docs images without an interactive session.
    const char* shot_path = std::getenv("CA_SHOT");
    if (shot_path) { opts.render_mode = RenderMode::Headless; }
    createInstance(opts, &instance);
    // Center the N^3 grid on the world origin so the orbit camera frames it: the camera sits back along
    // -z looking toward +z (through the origin), and the distance scales with n so the whole cube fits.
    const float3 grid_start = { -0.5f*n, -0.5f*n, -0.5f*n };
    setCameraPosition(instance, {0.f, 0.f, -2.2f*n});

    AllocHandle ping, pong, colormap;
    allocLinear(instance, (void**)&d1, sizeof(int) * n*n*n, &ping);
    allocLinear(instance, (void**)&d2, sizeof(int) * n*n*n, &pong);

    float4 *d_colors = nullptr;
    // Colormap: dead cells are fully transparent (we only care about living cells); living cells use
    // --cell-color at --opacity. index 0 = dead, index 1 = alive (the CA writes 0/1 into the state grid).
    float4 h_colors[2] = {
        { 0.f, 0.f, 0.f, 0.f },
        { cell_color.x, cell_color.y, cell_color.z, opacity },
    };
    unsigned int num_colors = std::size(h_colors);
    auto color_bytes = sizeof(float4) * num_colors;
    allocLinear(instance, (void**)&d_colors, color_bytes, &colormap);
    gpuErrchk(cudaMemcpy(d_colors, h_colors, color_bytes, cudaMemcpyHostToDevice));

    auto grid_layout = Layout::make(n, n, n);
    uint32_t index_count = n * n * n;
    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc{
        .type   = ViewType::Voxels,
        .domain = DomainType::Domain3D,
        .attributes  = {
            { AttributeType::Position, makeStructuredGrid(instance, grid_layout, grid_start) },
            { AttributeType::Color, {
                .source   = colormap,
                .size     = num_colors,
                .format   = FormatDescription::make<float4>(),
                .indexing = {
                    .source     = ping,
                    .size       = index_count,
                    .index_size = sizeof(int),
                }
            }}
        },
        .layout        = grid_layout,
        // default_color carries the living-cell color + opacity to the path tracer (raster voxels use
        // the colormap instead, so this is PT-only): albedo = cell color, alpha = --opacity.
        .default_color = { cell_color.x, cell_color.y, cell_color.z, opacity },
        .default_size  = .5f,
    };
    // Translucent volume (--opacity < 1): disable depth so living cubes behind others still blend in,
    // revealing interior cells. Opaque (opacity == 1) keeps normal depth testing.
    desc.depth_test = (opacity >= 0.999f);
    createView(instance, &desc, &v1);

    desc.attributes[AttributeType::Color].indexing.source = pong;
    desc.visible = false;
    createView(instance, &desc, &v2);

    // TODO CAMBIAR A 2D
    gpuErrchk(cudaMemcpy(d1, original, sizeof(int)*n*n*n, cudaMemcpyHostToDevice));
    printf("done: %f secs\n", omp_get_wtime() - t1);

    // Headless one-shot: render the initial state (v1/ping) offscreen and write it out, then exit.
    if (shot_path)
    {
        renderHeadless(instance, []{}, 1);
        saveFrame(instance, shot_path);
        printf("Wrote headless frame to %s (n=%ld, lod=%u)\n", shot_path, n, lod_cells);
        destroyInstance(instance);
        return EXIT_SUCCESS;
    }

    // ---- Performance HUD (top-right overlay), in the style of the particles-kmodal-3d sample ----
    // Proportional port: reads live metrics via the public getMetrics() and static device info via the
    // CUDA runtime (no NVML/powermon plumbing). Captured by value; getMetrics is read live each frame.
    struct CaHud {
        InstanceHandle instance;
        long     N;                 // fine grid resolution per axis
        unsigned M;                 // LOD coarsening factor (0 = off)
        unsigned long long fine;    // N^3 fine voxels
        unsigned long long coarse;  // M^3 coarse cubes drawn under LOD (== fine when off)
        int      seed;   float prob;   int block;   const char* mode;
        bool     fly;               // camera mode (for the controls hint)
        const char* light;          // --light-model name
        float    opacity;
        char     gpu_name[256];   char gpu_cc[64];   float gpu_total_gb;
    } cahud{};
    cahud.instance = instance;
    cahud.N = n;  cahud.M = lod_cells;
    cahud.fine   = (unsigned long long)n * n * n;
    cahud.coarse = lod_cells ? (unsigned long long)lod_cells * lod_cells * lod_cells : cahud.fine;
    cahud.seed = seed;  cahud.prob = prob;  cahud.block = B;  cahud.mode = map[modo];
    cahud.fly = fly;
    cahud.light = (light == VoxLight::Phong) ? "phong"
                : (light == VoxLight::PathTracing) ? "path-tracing" : "flat";
    cahud.opacity = opacity;
    {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
        snprintf(cahud.gpu_name, sizeof(cahud.gpu_name), "%s", prop.name);
        snprintf(cahud.gpu_cc, sizeof(cahud.gpu_cc), "%d (CC %d.%d)", dev, prop.major, prop.minor);
        cahud.gpu_total_gb = (float)((double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    // Window controls shared with the render-thread GUI callback: Ctrl+W quits, Space / the HUD button
    // toggle pause. The sim loop below polls these (paused = keep rendering, stop advancing the CA).
    std::atomic<bool> quit{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> step_once{false}; // advance exactly one CA step while paused
    setGuiCallback(instance, [cahud, &quit, &paused, &step_once]() {
        auto& io = ImGui::GetIO();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_W, /*repeat=*/false)) quit.store(true);
        if (ImGui::IsKeyPressed(ImGuiKey_Space, /*repeat=*/false))
            paused.store(!paused.load());
        // Right arrow steps one frame while paused (for inspecting a specific state).
        if (paused.load() && ImGui::IsKeyPressed(ImGuiKey_RightArrow, /*repeat=*/true))
            step_once.store(true);
        ImVec2 disp = io.DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(disp.x - 10.f, 10.f), ImGuiCond_Always, ImVec2(1.f, 0.f));
        ImGui::Begin("Performance", nullptr,
            ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_AlwaysAutoResize);
        PerformanceMetrics m = getMetrics(cahud.instance);
        #define HUD_ROW(label, ...) ImGui::TableNextRow(); \
            ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(label); \
            ImGui::TableSetColumnIndex(1); ImGui::Text(__VA_ARGS__)
        if (ImGui::BeginTable("scene", 2)) {
            HUD_ROW("GPU", "%s", cahud.gpu_name);
            HUD_ROW("Device", "%s", cahud.gpu_cc);
            HUD_ROW("VRAM", "%.1f GB", cahud.gpu_total_gb);
            HUD_ROW("VRAM used", "%.2f / %.2f GB", m.devmem.usage, m.devmem.budget);
            HUD_ROW("Grid", "%ld^3 (%llu voxels)", cahud.N, cahud.fine);
            if (cahud.M) {
                HUD_ROW("LOD", "%u^3 (%llu cubes, %.1fx fewer)",
                    cahud.M, cahud.coarse, (double)cahud.fine / (double)cahud.coarse);
            } else {
                HUD_ROW("LOD", "off");
            }
            HUD_ROW("Mode", "%s (B=%d)", cahud.mode, cahud.block);
            HUD_ROW("Light", "%s", cahud.light);
            HUD_ROW("Opacity", "%.2f", cahud.opacity);
            HUD_ROW("Seed", "%d", cahud.seed);
            HUD_ROW("Prob", "%.3f", cahud.prob);
            HUD_ROW("Camera", "%s", cahud.fly ? "Fly: WASD/QE, TAB=cursor" : "Orbit: drag/zoom/pan");
            ImGui::EndTable();
        }
        ImGui::Separator();
        if (ImGui::BeginTable("perf", 2)) {
            HUD_ROW("FPS", "%.1f", m.frame_rate);
            HUD_ROW("Compute", "%.2f ms", m.times.compute);
            HUD_ROW("Render", "%.2f ms", m.times.graphics);
            HUD_ROW("  Pipeline", "%.2f ms", m.times.pipeline);
            HUD_ROW("  GPU frame", "%.2f ms", m.times.gpu);
            ImGui::EndTable();
        }
        #undef HUD_ROW
        ImGui::Separator();
        if (ImGui::Button(paused.load() ? "Resume (Space)" : "Pause (Space)"))
            paused.store(!paused.load());
        ImGui::SameLine();
        ImGui::BeginDisabled(!paused.load());
        if (ImGui::Button("Step (Right)")) step_once.store(true); // one step while paused
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Quit (Ctrl+W)")) quit.store(true);
        ImGui::End();

        // Top-left panel: how to interact.
        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), ImGuiCond_Always, ImVec2(0.f, 0.f));
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_NoResize   | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::TextUnformatted("Controls");
        ImGui::Separator();
        if (cahud.fly)
        {
            ImGui::TextUnformatted("Move        : W A S D");
            ImGui::TextUnformatted("Up / Down   : Q E");
            ImGui::TextUnformatted("Look        : mouse");
            ImGui::TextUnformatted("Free cursor : TAB");
        }
        else
        {
            ImGui::TextUnformatted("Rotate : left-drag");
            ImGui::TextUnformatted("Zoom   : right-drag");
            ImGui::TextUnformatted("Pan    : middle-drag");
        }
        ImGui::Separator();
        ImGui::Text("Pause/Resume : Space  (%s)", paused.load() ? "paused" : "running");
        ImGui::TextUnformatted("Step (paused): Right arrow");
        ImGui::TextUnformatted("Quit         : Ctrl+W");
        ImGui::End();
    });

    dim3 block(B, B, B);
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y, (n+block.z-1)/block.z);

    if (modo == 1)
    {
        // GPU: async render thread + this driver loop (same structure as particles-kmodal-3d). The CA
        // advances one step per frame with no Enter prompts; after `steps` the loop keeps rendering the
        // final state until Ctrl+W or the window closes. Space / the HUD button pause (keep rendering,
        // stop advancing). prepareViews/updateViews are the CUDA<->Vulkan interop handshake; calling them
        // with no kernel between (paused or finished) simply re-presents the current state, frame-paced.
        printf("Running %d GPU steps (Space=pause, Ctrl+W=quit)...\n", steps); fflush(stdout);
        displayAsync(instance);
        int step_i = 0;
        while (isRunning(instance) && !quit.load())
        {
            prepareViews(instance);
            // Advance when running, or exactly once per Step request while paused.
            bool advance = step_i < steps
                && (!paused.load() || step_once.exchange(false));
            if (advance)
            {
                kernel_CA3D<<<grid, block>>>(n, d1, d2);
                gpuErrchk(cudaPeekAtLastError());

                // CA_LOD_CHECK: one-shot validation of the max-pool formula the LOD vertex shader uses.
                // Pools the freshly computed fine state (d2) with the tested reference kernel AND a CPU
                // max-pool and asserts they agree. Guarded by CA_LOD_CHECK; runs once (first step).
                if (step_i == 0 && lod_cells > 0 && std::getenv("CA_LOD_CHECK"))
                {
                    const uint32_t N = (uint32_t)n, M = lod_cells;
                    if (M > 0 && M < N)
                    {
                        gpuErrchk(cudaDeviceSynchronize());
                        const size_t nc = (size_t)N*N*N, mc = (size_t)M*M*M;
                        std::vector<int> hfine(nc);
                        gpuErrchk(cudaMemcpy(hfine.data(), d2, nc*sizeof(int), cudaMemcpyDeviceToHost));

                        auto lo = [&](uint32_t c){ return (uint32_t)((uint64_t)c*N/M); };
                        auto hi = [&](uint32_t c){ return (uint32_t)(((uint64_t)c+1)*N/M); };
                        std::vector<int> cpu(mc, 0);
                        for (uint32_t cz = 0; cz < M; ++cz)
                        for (uint32_t cy = 0; cy < M; ++cy)
                        for (uint32_t cx = 0; cx < M; ++cx)
                        {
                            int mx = 0;
                            for (uint32_t z = lo(cz); z < hi(cz); ++z)
                            for (uint32_t y = lo(cy); y < hi(cy); ++y)
                            for (uint32_t x = lo(cx); x < hi(cx); ++x)
                                mx = std::max(mx, hfine[(size_t)x + N*((size_t)y + (size_t)N*z)]);
                            cpu[(size_t)cx + M*((size_t)cy + (size_t)M*cz)] = mx;
                        }

                        int *dcoarse = nullptr;
                        gpuErrchk(cudaMalloc(&dcoarse, mc*sizeof(int)));
                        mimir::voxelPoolMax(d2, N, dcoarse, M, 0);
                        gpuErrchk(cudaDeviceSynchronize());
                        std::vector<int> gpu(mc);
                        gpuErrchk(cudaMemcpy(gpu.data(), dcoarse, mc*sizeof(int), cudaMemcpyDeviceToHost));
                        cudaFree(dcoarse);

                        size_t bad = 0;
                        for (size_t k = 0; k < mc; ++k) if (cpu[k] != gpu[k]) ++bad;
                        printf("CA_LOD_CHECK: %s (N=%u M=%u, %zu coarse cells, %zu mismatch)\n",
                            bad ? "FAIL" : "OK", N, M, mc, bad);
                        fflush(stdout);
                    }
                }

                // Show the freshly written buffer: v1 renders ping, v2 renders pong; toggle which is
                // visible and swap the ping/pong pointers so the next step writes the other buffer.
                toggleVisibility(v1);
                toggleVisibility(v2);
                std::swap(d1, d2);
                ++step_i;
            }
            updateViews(instance);
        }
        exit(instance);
    }
    else
    {
        // CPU reference solver (console only, no live render): run all steps without pausing.
        int *CPUd2 = new int[n*n*n];
        for (int i = 0; i < steps; ++i)
        {
            t1 = omp_get_wtime();
            openmp_CA3D(n, original, CPUd2);
            printf("[CPU] step=%d done: %f s\n", i, omp_get_wtime() - t1);
            std::swap(original, CPUd2);
        }
        delete[] CPUd2;
    }
    printf("Finished running all steps\n");
    destroyInstance(instance);
}
