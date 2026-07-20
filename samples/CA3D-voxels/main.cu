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
#include <sys/sysinfo.h> // sysinfo (total RAM)

#include "tools.h"
#include "kernel3D.cuh"
#include "openmp3D.h"

#include <mimir/mimir.hpp>
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
    printf("  --steps-per-frame K   advance K CA steps per rendered frame (default 1; faster evolution)\n");
    printf("  --fps K               cap the frame rate at K (default 60; K <= 0 = uncapped)\n");
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
    int      steps_per_frame = 1;                 // CA steps advanced per rendered frame
    int      fps_cap = 60;                        // frame-rate cap (<= 0 => uncapped)
    for (int i = 8; i < argc; ++i)
    {
        std::string a = argv[i];
        if      (a == "--lod" && i + 1 < argc) lod_cells = (unsigned)std::stoul(argv[++i]);
        else if (a == "--fly")                 fly = true;
        else if (a == "--opacity" && i + 1 < argc) opacity = std::stof(argv[++i]);
        else if (a == "--steps-per-frame" && i + 1 < argc) steps_per_frame = std::max(1, atoi(argv[++i]));
        else if (a == "--fps" && i + 1 < argc) fps_cap = atoi(argv[++i]);
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
    if (fly) { opts.camera_move_speed = 1.5f * n; } // scale WASD speed to the grid (was too slow)
    // Living-voxel shading. flat = unlit, phong = lit cubes, path-tracing = ray-traced boxes (RT GPU).
    opts.light_model = (light == VoxLight::Phong)       ? LightModel::Phong
                     : (light == VoxLight::PathTracing) ? LightModel::PathTracing
                                                        : LightModel::None;
    opts.background_color = { bg_color.x, bg_color.y, bg_color.z, 1.f };
    // Diagonal key light from behind-upper-left of the camera (both cameras view the +z faces, see
    // below), so those faces are front-lit -- clear directional shading in phong and path tracing.
    opts.light_pos = { -0.4f, 0.6f, 0.7f };
    // Frame-rate cap: --fps K limits rendering to K fps (default 60); K <= 0 uncaps it.
    if (fps_cap > 0) { opts.present.enable_fps_limit = true;  opts.present.target_fps = fps_cap; }
    else             { opts.present.enable_fps_limit = false; }
    // Built-in FPS/frame-time overlay (F2 toggles it). No ImGui code in this sample.
    opts.show_hud = true;
    if (light == VoxLight::PathTracing)
    {
        // A few samples/bounces per frame keep the live path-traced volume readable (it also keeps
        // accumulating across static frames). Transmission (--opacity < 1) is noisier, so lean higher.
        opts.pt_samples_per_pixel = (opacity < 1.f) ? 8 : 4;
        opts.pt_max_bounces       = 3;
    }
    // CA_SHOT=<path>: render one offscreen frame of the initial state to a PPM and exit (no window).
    // Handy for eyeballing the LOD framing / capturing docs images without an interactive session.
    const char* shot_path = std::getenv("CA_SHOT");
    if (shot_path) { opts.render_mode = RenderMode::Headless; }
    createInstance(opts, &instance);
    // Center the N^3 grid on the world origin and aim the camera at it from the +z side (looking -z),
    // so both orbit and fly see the +z faces the diagonal light front-lights. setCameraLookAt takes
    // eye/target/up directly -- no per-control-mode sign juggling to reverse-engineer.
    const float3 grid_start = { -0.5f*n, -0.5f*n, -0.5f*n };
    setCameraLookAt(instance, { 0.f, 0.f, 2.2f*n }, { 0.f, 0.f, 0.f }, { 0.f, 1.f, 0.f });

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

    // Scene summary + controls to the console. The live FPS/frame-time overlay is drawn by mimir
    // itself (opts.show_hud, F2 toggles it) -- this sample links no ImGui and writes no GUI code.
    {
        unsigned long long fine = (unsigned long long)n * n * n;
        const char* light_name = (light == VoxLight::Phong) ? "phong"
            : (light == VoxLight::PathTracing) ? "path-tracing" : "flat";
        printf("Scene : grid %ld^3 (%llu voxels), light=%s, opacity=%.2f", n, fine, light_name, opacity);
        if (lod_cells) { printf(", LOD %u^3 (%.1fx fewer cubes)", lod_cells,
            (double)fine / ((double)lod_cells * lod_cells * lod_cells)); }
        printf("\nControls: %s | Space=pause  .=step  F2=HUD  Ctrl+W=quit\n",
            fly ? "WASD/QE move, mouse look, TAB frees cursor"
                : "left-drag rotate, right-drag zoom, middle-drag pan");
    }

    dim3 block(B, B, B);
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y, (n+block.z-1)/block.z);

    if (modo == 1)
    {
        // GPU: async render thread + this driver loop (same structure as particles-kmodal-3d). The CA
        // advances one step per frame with no Enter prompts; after `steps` the loop keeps rendering the
        // final state until Ctrl+W or the window closes. Pause/step/quit are handled by mimir itself
        // (Space pauses, '.' steps, Ctrl+W quits) -- this loop just reads isPaused()/shouldStep() to
        // decide how many CA steps to advance. prepareViews/updateViews are the CUDA<->Vulkan interop
        // handshake; calling them with no kernel between simply re-presents the current state.
        printf("Running %d GPU steps (Space=pause, .=step, Ctrl+W=quit)...\n", steps); fflush(stdout);
        displayAsync(instance);
        int step_i = 0;
        while (isRunning(instance))
        {
            prepareViews(instance);
            // Advance up to steps_per_frame CA steps this frame while running; when paused, exactly one
            // step per queued step request. All steps run in one interop section (one render per frame).
            int budget = isPaused(instance) ? (shouldStep(instance) ? 1 : 0) : steps_per_frame;
            for (int s = 0; s < budget && step_i < steps; ++s)
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
