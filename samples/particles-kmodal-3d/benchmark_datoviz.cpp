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

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <datoviz.h>

#include "axes_gizmo.hpp"
#include "kmodal_sim.cuh"
#include "nvmlPower.hpp"
#include "validation.hpp"

using clk = std::chrono::high_resolution_clock;
static inline double ms_since(clk::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// datoviz's vec3 is a raw float[3] (cglm), which can't be a std::vector element (not assignable).
// Vec3 is a layout-identical wrapper we can store in vectors and cast to vec3* at the datoviz calls.
using Vec3 = std::array<float, 3>;

// ---------------------------------------------------------------------------
// Input / output structs
// ---------------------------------------------------------------------------

// datoviz raster geometry selected by --geometry. Point/Disc/Sphere are single-vertex point
// sprites (equal vertex cost); they differ only in the fragment shader and depth behavior:
//   Point  = dvz_point:  antialiased filled disc, no lighting, no depth write (early-Z intact) -- cheapest
//   Disc   = dvz_marker: SDF disc + AA,           no lighting, no depth write (early-Z intact)
//   Sphere = dvz_sphere: ray-sphere + Phong + per-fragment gl_FragDepth (early-Z defeated)
//   Mesh   = dvz_mesh:   real lit triangle icospheres, mimir's phong-mesh geometry. datoviz has no
//                        per-instance streaming, so all N spheres live in ONE merged mesh and every
//                        vertex position is re-expanded and re-uploaded each frame (N*V verts). This
//                        is datoviz's most efficient mesh path and is inherently far costlier than
//                        mimir's zero-copy instanced draw -- it exists to quantify exactly that gap.
enum class DvzMode { Point, Disc, Sphere, Mesh };

struct PointsInput {
    int          win_width  = 1920;
    int          win_height = 1080;
    PointsParams pts        = {};
    int          iter_count = 1000000;
    bool         vsync      = true;  // real display vsync (DVZ_CANVAS_FLAGS_VSYNC)
    bool         display    = true;
    float        size_px    = 5.f;   // marker size in pixels (same as benchmark_mimir --size)
    DvzMode      mode       = DvzMode::Disc; // render geometry (see DvzMode)
    uint32_t     subdiv     = 2;             // icosphere subdivisions for phong-mesh (--subdiv), matches mimir
    float3       background = { 0.f, 0.f, 0.f };       // panel background color (--background)
    float3       pcolor     = { 1.f, 1.f, 1.f };       // particle color (--pcolor)
    bool         fly        = false;                   // --fly: FPS fly camera instead of arcball
    bool         axes       = false;                   // --axes: draw the XYZ orientation triad
    float        timeout_s  = 0.f;                     // --timeout SECS: wall-clock cap (0 = all iters)
};

// Parse --background/--pcolor as "G" (grey level) or "R,G,B" in [0,1]. Mirrors benchmark_mimir.
static float3 parseColor(const std::string& v)
{
    float r = 0.f, g = 0.f, b = 0.f;
    if (std::sscanf(v.c_str(), "%f,%f,%f", &r, &g, &b) == 3) { return { r, g, b }; }
    float grey = std::stof(v);
    return { grey, grey, grey };
}

// Convert a [0,1] float3 color into datoviz's 8-bit RGBA (cvec4), opaque.
static void toDvzColor(float3 c, DvzColor out)
{
    auto q = [](float x) -> uint8_t {
        float v = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
        return (uint8_t)(v * 255.f + 0.5f);
    };
    out[0] = q(c.x); out[1] = q(c.y); out[2] = q(c.z); out[3] = 255;
}

// Build a unit icosphere (vertex positions double as normals) using the SAME midpoint-subdivision
// geometry as mimir's ensureSphereMesh(), so datoviz's mesh spheres match mimir's phong-mesh
// triangle-for-triangle. subdiv 0/1/2 -> 12/42/162 verts, 20/80/320 faces.
static void buildIcosphere(uint32_t subdiv, std::vector<Vec3>& out_verts, std::vector<DvzIndex>& out_idx)
{
    struct V3 { float x, y, z; };
    auto norm = [](V3 v) { float l = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); return V3{v.x/l, v.y/l, v.z/l}; };
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<V3> verts = {
        {-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0}, {0,-1,t},{0,1,t},
        {0,-1,-t},{0,1,-t}, {t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1},
    };
    for (auto& v : verts) { v = norm(v); }
    struct Tri { uint32_t a, b, c; };
    std::vector<Tri> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11}, {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9}, {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    };
    for (uint32_t s = 0; s < subdiv; ++s)
    {
        std::unordered_map<uint64_t, uint32_t> cache;
        auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
            uint64_t key = a < b ? ((uint64_t)a << 32 | b) : ((uint64_t)b << 32 | a);
            auto it = cache.find(key);
            if (it != cache.end()) { return it->second; }
            V3 m = norm(V3{ (verts[a].x+verts[b].x)*0.5f, (verts[a].y+verts[b].y)*0.5f, (verts[a].z+verts[b].z)*0.5f });
            uint32_t idx = (uint32_t)verts.size(); verts.push_back(m); cache.emplace(key, idx); return idx;
        };
        std::vector<Tri> next; next.reserve(faces.size() * 4);
        for (const auto& f : faces) {
            uint32_t a = midpoint(f.a, f.b), b = midpoint(f.b, f.c), c = midpoint(f.c, f.a);
            next.push_back({f.a,a,c}); next.push_back({f.b,b,a}); next.push_back({f.c,c,b}); next.push_back({a,b,c});
        }
        faces.swap(next);
    }
    out_verts.resize(verts.size());
    for (size_t i = 0; i < verts.size(); ++i) { out_verts[i][0]=verts[i].x; out_verts[i][1]=verts[i].y; out_verts[i][2]=verts[i].z; }
    out_idx.clear(); out_idx.reserve(faces.size() * 3);
    for (const auto& f : faces) { out_idx.push_back(f.a); out_idx.push_back(f.b); out_idx.push_back(f.c); }
}

// Expand N sphere centers into N*V vertex positions (template scaled by radius + center). This is
// the per-frame "pack" datoviz's mesh path forces because it has no per-instance streaming.
static void expandMesh(const std::vector<Vec3>& tmpl, const vec3* centers, uint32_t n, float radius, Vec3* out)
{
    const uint32_t V = (uint32_t)tmpl.size();
    for (uint32_t i = 0; i < n; ++i)
    {
        const float cx = centers[i][0], cy = centers[i][1], cz = centers[i][2];
        Vec3* dst = out + (size_t)i * V;
        for (uint32_t k = 0; k < V; ++k)
        {
            dst[k][0] = cx + tmpl[k][0] * radius;
            dst[k][1] = cy + tmpl[k][1] * radius;
            dst[k][2] = cz + tmpl[k][2] * radius;
        }
    }
}

// datoviz is the raster baseline (DESIGN_pathtracing.md §7) and has one primitive per geometry, so
// only mimir's --geometry axis selects anything here: sprite -> flat disc markers, impostor -> lit
// sphere impostors, mesh -> lit triangle icospheres (one merged mesh). datoviz's raw dvz_point has
// no mimir equivalent, so it keeps the extra 'point' spelling. --shading path-traced is unavailable.
static DvzMode parseGeometryMode(const std::string& v)
{
    if (v == "point")                     return DvzMode::Point;
    if (v == "sprite" || v == "flat" || v == "none") return DvzMode::Disc;
    if (v == "impostor" || v == "phong" || v == "sphere") return DvzMode::Sphere;
    if (v == "mesh" || v == "phong-mesh") return DvzMode::Mesh;
    if (v == "path-traced" || v == "path-tracing")
    {
        fprintf(stderr,
            "datoviz is the raster baseline and cannot path trace; "
            "run benchmark_mimir --shading path-traced instead.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Unknown geometry '%s' (use point|sprite|impostor|mesh)\n", v.c_str());
    exit(EXIT_FAILURE);
}

struct HudData {
    unsigned int points;
    int          frame;
    float        fps;
    float        compute_ms;
    float        pack_ms;      // phong-mesh: center -> N*V vertex expansion; 0 for point-sprite modes
    float        d2h_ms;       // cudaMemcpy Device->Host (PCIe DMA)
    float        h2h_ms;       // dvz_marker_position (pinned host -> datoviz internal heap copy)
    float        render_ms;    // dvz_scene_step: H2D + D2D + draw (inseparable)
    float        gpu_watts;
    char         gpu_name[256];
    char         gpu_device[64];   // "N (CC major.minor)"
    float        gpu_total_gb;  // total VRAM (NVML)
    // VRAM used (NVML) dismembered so the sub-parts sum to it. External/CUDA are anchored on
    // measured NVML checkpoints at startup; Render is computed; Graphics is the remainder.
    float        vram_used_mb;      // VRAM in use right now (NVML, all processes on the GPU)
    float        vram_external_mb;  // measured: NVML used before we touch the GPU (other processes)
    float        cuda_ctx_mb;       // measured: CUDA context reservation (checkpoint delta - buf)
    float        buf_mb;            // computed: our CUDA sim device buffers (d_pos + rng + clusters)
    float        render_mb;         // computed: our datoviz render geometry (mode-dependent)
    float        vulkan_mb;         // computed: Vulkan render targets (swapchain + depth + staging)
    uint32_t     seed;
    unsigned int k;
    float        epsilon;
    bool         fly;          // --fly active: show the fly-camera controls hint
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
    int                iters;  // iterations executed; divide the time TOTALS by this for per-frame averages
};

struct DatovizContext {
    DvzApp*     app;
    DvzBatch*   batch;
    DvzScene*   scene;
    DvzFigure*  figure;
    DvzPanel*   panel;
    DvzArcball* arcball;   // 3D interactivity when --fly is off (null otherwise)
    DvzFly*     fly;       // FPS fly camera when --fly is on (null otherwise)
    DvzVisual*  visual;
    // phong-mesh only: unit icosphere template (positions == normals), shared by all N spheres.
    std::vector<Vec3> tmpl_verts;
    // Device-side geometry we hand datoviz (the per-vertex/-index attribute buffers), in bytes.
    // This is the "extra VRAM" the chosen render mode costs; the mesh mode dwarfs the others.
    double render_bytes = 0.0;
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
    // Column names carry units; time columns are TOTALS over the run in seconds, memory in GB, power
    // in W, energy in J. pipeline_time_s is 0 (datoviz has no render-pass timing API). iters is placed
    // with the experiment settings (next to N). Leading columns match benchmark_mimir for comparison.
    printAligned({
        {"mode",            mode},
        {"windowres",       resolution},
        {"N",               sd((int)input.pts.count)},
        {"iters",           sd(result.iters)},
        {"seed",            su(input.pts.seed)},
        {"k",               su(input.pts.k)},
        {"epsilon",         sf(input.pts.epsilon)},
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
        {"pack_time_s",     sf(lib.transfer.pack)},
        {"d2h_time_s",      sf(lib.transfer.d2h)},
        {"h2h_time_s",      sf(lib.transfer.h2h)},
    });
    printf("%s,%s,%u,%d,%u,%u,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
        mode.c_str(), resolution.c_str(),
        input.pts.count, result.iters, input.pts.seed, input.pts.k, input.pts.epsilon,
        lib.frame_rate, lib.times.compute, lib.times.pipeline, lib.times.graphics,
        lib.devmem.usage, lib.devmem.budget,
        gpu.average_power, gpu.total_energy, gpu.total_time,
        nvml.free, nvml.reserved, nvml.total, nvml.used,
        lib.transfer.pack, lib.transfer.d2h, lib.transfer.h2h);
}

// ---------------------------------------------------------------------------
// Keyboard callback — Ctrl+W closes the window
// ---------------------------------------------------------------------------

// HUD visibility, toggled with F1 (clean-viewport screenshots). Written by the keyboard
// callback, read by the GUI callback.
static std::atomic<bool> g_show_hud{true};

static void keyCallback(DvzApp* /*app*/, DvzId /*window_id*/, DvzKeyboardEvent* ev)
{
    if (ev->type == DVZ_KEYBOARD_EVENT_PRESS
        && ev->key  == DVZ_KEY_W
        && (ev->mods & DVZ_KEY_MODIFIER_CONTROL))
    {
        auto* flag = static_cast<std::atomic<bool>*>(ev->user_data);
        flag->store(true);
    }
    if (ev->type == DVZ_KEYBOARD_EVENT_PRESS && ev->key == DVZ_KEY_F1)
    {
        g_show_hud.store(!g_show_hud.load());
    }
}

// ---------------------------------------------------------------------------
// HUD callback (datoviz GUI — called from within dvz_scene_step)
// ---------------------------------------------------------------------------

static void hudCallback(DvzApp* /*app*/, DvzId /*canvas_id*/, DvzGuiEvent* ev)
{
    if (!g_show_hud.load()) { return; }
    auto* hud = static_cast<HudData*>(ev->user_data);
    // Borderless overlay in the top-right corner, matching benchmark_mimir's Performance HUD.
    dvz_gui_corner(DVZ_DIALOG_CORNER_TOP_RIGHT, (vec2){10, 10});
    dvz_gui_begin("Datoviz - particles-kmodal-3d", DVZ_DIALOG_FLAGS_OVERLAY);

    // Hardware / run parameters (same fields and order as benchmark_mimir).
    dvz_gui_text("GPU        %s",       hud->gpu_name);
    dvz_gui_text("Device     %s",       hud->gpu_device);
    dvz_gui_text("VRAM       %.1f GB",  hud->gpu_total_gb);
    dvz_gui_text("Points     %u",       hud->points);
    dvz_gui_text("Seed       %u",       hud->seed);
    dvz_gui_text("Modes k    %u",       hud->k);
    dvz_gui_text("Epsilon    %.4f",     hud->epsilon);
    // VRAM used (NVML, whole GPU) fully dismembered; the six sub-lines sum to it by construction.
    // External + CUDA ctx are anchored on measured NVML checkpoints (before GPU touch / after
    // forcing the CUDA context); CUDA buf + Render + Vulkan are computed from known sizes; Datoviz
    // is the remainder = datoviz's own device structures (buffer pools, uniforms, pipelines, imgui).
    float datoviz_mb = hud->vram_used_mb - hud->vram_external_mb - hud->cuda_ctx_mb
                     - hud->buf_mb - hud->render_mb - hud->vulkan_mb;
    dvz_gui_text("VRAM used         %.0f MB", hud->vram_used_mb);
    dvz_gui_text("  External procs  %.0f MB", hud->vram_external_mb); // other processes (measured)
    dvz_gui_text("  CUDA context    %.0f MB", hud->cuda_ctx_mb);      // CUDA runtime reserve (measured)
    dvz_gui_text("  CUDA buffers    %.0f MB", hud->buf_mb);           // our sim device buffers (computed)
    dvz_gui_text("  Render geometry %.0f MB", hud->render_mb);        // our geometry (computed)
    dvz_gui_text("  Vulkan targets  %.0f MB", hud->vulkan_mb);        // swapchain + depth + staging
    dvz_gui_text("  Datoviz structs %.0f MB", datoviz_mb);           // datoviz buffer pools/pipelines
    dvz_gui_text("");

    // Per-frame timing. Unlike mimir (zero-copy, so Transfer is N/A) datoviz actually pays the
    // Device->Host->datoviz transfer, so those sub-metrics carry real values here. Conversely the
    // datoviz Render is a single opaque dvz_scene_step (H2D staging + D2D upload + draw are
    // inseparable), so it has no GPU-frame/Wait/Record/Submit breakdown like mimir's Render.
    dvz_gui_text("Frame      %d",        hud->frame);
    dvz_gui_text("FPS        %.1f",      hud->fps);
    dvz_gui_text("Compute    %.2f ms",   hud->compute_ms);
    dvz_gui_text("Transfer   %.2f ms",   hud->pack_ms + hud->d2h_ms + hud->h2h_ms);
    dvz_gui_text("    Pack   %.2f ms",   hud->pack_ms);
    dvz_gui_text("    D2H    %.2f ms",   hud->d2h_ms);
    dvz_gui_text("    H2H    %.2f ms",   hud->h2h_ms);
    dvz_gui_text("Render     %.2f ms",   hud->render_ms);
    dvz_gui_text("    (H2D + D2D + draw, inseparable)");
    dvz_gui_text("Power      %.1f W",   hud->gpu_watts);

    if (hud->fly)
    {
        dvz_gui_text("");
        dvz_gui_text("Fly: arrows move, drag look, wheel zoom");
    }
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
    // --fly selects datoviz's FPS fly camera (matching benchmark_mimir --fly); otherwise the
    // default arcball trackball. FIXED_UP keeps a world up-vector, like mimir's fly camera.
    if (input.fly) { ctx.fly = dvz_panel_fly(ctx.panel, DVZ_FLY_FLAGS_FIXED_UP); }
    else           { ctx.arcball = dvz_panel_arcball(ctx.panel, 0); }

    // Panel background (--background): fill all four corners with the same color.
    DvzColor bg; toDvzColor(input.background, bg);
    DvzColor corners[4] = { {bg[0],bg[1],bg[2],bg[3]}, {bg[0],bg[1],bg[2],bg[3]},
                            {bg[0],bg[1],bg[2],bg[3]}, {bg[0],bg[1],bg[2],bg[3]} };
    dvz_panel_background(ctx.panel, corners);

    // Particle color (--pcolor), applied uniformly to every point.
    DvzColor pc; toDvzColor(input.pcolor, pc);
    std::vector<DvzColor> colors(n);
    for (unsigned int i = 0; i < n; i++)
    {
        colors[i][0] = pc[0]; colors[i][1] = pc[1]; colors[i][2] = pc[2]; colors[i][3] = pc[3];
    }

    if (input.mode == DvzMode::Sphere)
    {
        // Lit sphere impostors (ray-sphere point sprites with Phong lighting and
        // per-fragment depth) — same technique as mimir's Sphere3D marker mode.
        //
        // Size matching (the divisor is /100, not /50): mimir's default_size is the sphere
        // *world radius* (size/100), drawn by a geometry-shader impostor billboard that projects
        // geometrically. datoviz's dvz_sphere_size feeds graphics_sphere.frag, where the sphere
        // fills the whole point sprite (discard if dist>1), so the on-screen diameter is
        // gl_PointSize = s * viewport.y * proj[1][1] / w. That formula uses the FULL viewport
        // height where a correct projection uses viewport.y/2, so datoviz draws a sphere exactly
        // 2x larger on screen than mimir for the same world size. Halving it (size/100 instead of
        // size/50) cancels that factor and makes datoviz match mimir 1:1.
        ctx.visual = dvz_sphere(ctx.batch, DVZ_SPHERE_FLAGS_LIGHTING);
        dvz_sphere_alloc(ctx.visual, n);
        // Directional sun (w=0 -> lighting.glsl normalizes light.pos and ignores
        // distance) coming diagonally from behind-and-above the camera. Same world-space
        // convention and SAME normalized vector as benchmark_mimir's opts.light_pos, so
        // both samples are lit by one shared sun instead of datoviz's default light.
        vec4 sun_dir = { -0.4082f, 0.4082f, 0.8165f, 0.f }; // normalize({-1, 1, 2}), w=0
        dvz_sphere_light_pos(ctx.visual, 0, sun_dir);
        dvz_sphere_position(ctx.visual, 0, n, (vec3*)initial_pos3, 0);
        std::vector<float> sizes(n, input.size_px / 100.f);
        dvz_sphere_size(ctx.visual, 0, n, sizes.data(), 0);
        dvz_sphere_color(ctx.visual, 0, n, colors.data(), 0);
        // 1 vertex/particle: position(12) + size(4) + color(4).
        ctx.render_bytes = (double)n * (sizeof(vec3) + sizeof(float) + sizeof(DvzColor));
    }
    else if (input.mode == DvzMode::Point)
    {
        // Cheapest path: raw point sprites (dvz_point). The fragment shader is just an
        // antialiased filled disc (no SDF shape machinery, no lighting, no gl_FragDepth), so
        // early-Z stays intact. Size is gl_PointSize in pixels, same as the disc marker.
        ctx.visual = dvz_point(ctx.batch, 0);
        dvz_point_alloc(ctx.visual, n);
        dvz_point_position(ctx.visual, 0, n, (vec3*)initial_pos3, 0);
        std::vector<float> sizes(n, input.size_px);
        dvz_point_size(ctx.visual, 0, n, sizes.data(), 0);
        dvz_point_color(ctx.visual, 0, n, colors.data(), 0);
        // 1 vertex/particle: position(12) + size(4) + color(4).
        ctx.render_bytes = (double)n * (sizeof(vec3) + sizeof(float) + sizeof(DvzColor));
    }
    else if (input.mode == DvzMode::Mesh)
    {
        // Real lit triangle icospheres — mimir's phong-mesh geometry. datoviz's mesh visual is
        // per-vertex only (no instancing), so we build ONE merged mesh of N icospheres: static
        // normals/indices/colors uploaded once here, positions re-expanded and re-uploaded every
        // frame (the loop's pack + h2h stages). radius = size/100, matching mimir's default_size.
        std::vector<DvzIndex> tmpl_idx;
        buildIcosphere(input.subdiv, ctx.tmpl_verts, tmpl_idx);
        const uint32_t V  = (uint32_t)ctx.tmpl_verts.size();
        const uint32_t I  = (uint32_t)tmpl_idx.size();
        const uint32_t un = n;
        const float radius = input.size_px / 100.f;

        fprintf(stderr, "phong-mesh: icosphere subdiv %u (%u verts, %u tris) x %u spheres = "
            "%.1f M verts; per-frame position upload ~%.0f MB\n",
            input.subdiv, V, I / 3, un, (double)un * V / 1e6,
            (double)un * V * sizeof(vec3) / (1024.0 * 1024.0));

        // DVZ_VISUAL_FLAGS_INDEXED is REQUIRED for datoviz to honor the index buffer; without it
        // the mesh is drawn non-indexed (vertices in order), mangling every triangle.
        ctx.visual = dvz_mesh(ctx.batch, DVZ_MESH_FLAGS_LIGHTING | DVZ_VISUAL_FLAGS_INDEXED);
        dvz_mesh_alloc(ctx.visual, un * V, un * I);

        // Device geometry: N spheres each get their own V verts / I indices (datoviz can't
        // instance). The vertex buffer is baked into the FULL interleaved DvzMeshColorVertex, so we
        // pay for every attribute the mesh shader declares even though we only fill pos/normal/color:
        //   pos(12)+normal(12)+color(4)+value(4)+d_left(12)+d_right(12)+contour(4) = 60 B/vertex.
        // (The persistent staging copy datoviz keeps is host RAM -- VMA_MEMORY_USAGE_CPU_ONLY -- so
        // it is NOT counted here; buffers are not duplicated per swapchain image either.)
        const uint32_t kMeshVertexStride = 60; // sizeof(DvzMeshColorVertex)
        ctx.render_bytes = (double)un * V * kMeshVertexStride
                         + (double)un * I * sizeof(DvzIndex);

        // Static per-vertex data, uploaded once (only positions change per frame). Each temporary
        // is scoped so it frees before the next big allocation, capping peak host memory.
        {
            std::vector<DvzIndex> idx((size_t)un * I);
            for (uint32_t s = 0; s < un; ++s)
                for (uint32_t k = 0; k < I; ++k) idx[(size_t)s * I + k] = tmpl_idx[k] + s * V;
            dvz_mesh_index(ctx.visual, 0, un * I, idx.data(), 0);
        }
        {
            std::vector<Vec3> nrm((size_t)un * V);
            for (uint32_t s = 0; s < un; ++s)
                for (uint32_t k = 0; k < V; ++k) nrm[(size_t)s * V + k] = ctx.tmpl_verts[k];
            dvz_mesh_normal(ctx.visual, 0, un * V, (vec3*)nrm.data(), 0);
        }
        {
            std::vector<DvzColor> mcol((size_t)un * V);
            for (auto& c : mcol) { c[0]=pc[0]; c[1]=pc[1]; c[2]=pc[2]; c[3]=pc[3]; }
            dvz_mesh_color(ctx.visual, 0, un * V, mcol.data(), 0);
        }

        vec4 sun_dir = { -0.4082f, 0.4082f, 0.8165f, 0.f }; // same shared sun as Sphere mode
        dvz_mesh_light_pos(ctx.visual, 0, sun_dir);

        {
            std::vector<Vec3> init_pos((size_t)un * V);
            expandMesh(ctx.tmpl_verts, (vec3*)initial_pos3, un, radius, init_pos.data());
            dvz_mesh_position(ctx.visual, 0, un * V, (vec3*)init_pos.data(), 0);
        }
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
        // 1 vertex/particle: position(12) + size(4) + color(4).
        ctx.render_bytes = (double)n * (sizeof(vec3) + sizeof(float) + sizeof(DvzColor));
    }

    dvz_panel_visual(ctx.panel, ctx.visual, 0);

    // --axes: world XYZ orientation triad with letter labels (see axes_gizmo.hpp; shared with
    // benchmark_mimir so both draw the identical gizmo), rendered with dvz_segment.
    if (input.axes)
    {
        const auto segs = makeAxesGizmo();
        const uint32_t n_segs = (uint32_t)segs.size();
        std::vector<Vec3>     ini(n_segs), ter(n_segs);
        std::vector<DvzColor> col(n_segs);
        // Screen-space pixel shifts (dx0,dy0,dx1,dy1): billboard the letter strokes around
        // their anchors (0 on the axis lines). See axes_gizmo.hpp.
        std::vector<std::array<float, 4>> shifts(n_segs);
        std::vector<float>    widths(n_segs, 3.f); // same boldness as the mimir gizmo's linewidth
        for (uint32_t i = 0; i < n_segs; ++i)
        {
            const auto& s = segs[i];
            ini[i]    = { s.ax, s.ay, s.az };
            ter[i]    = { s.bx, s.by, s.bz };
            shifts[i] = { s.sax, s.say, s.sbx, s.sby };
            toDvzColor({ s.r, s.g, s.b }, col[i]);
        }
        DvzVisual* axes = dvz_segment(ctx.batch, 0);
        // Overlay: no depth test, so the triad and its labels stay readable even when a
        // particle cluster sits right on an axis (same as the mimir side's depth_test=false).
        dvz_visual_depth(axes, DVZ_DEPTH_TEST_DISABLE);
        dvz_segment_alloc(axes, n_segs);
        dvz_segment_position(axes, 0, n_segs, (vec3*)ini.data(), (vec3*)ter.data(), 0);
        dvz_segment_shift(axes, 0, n_segs, (vec4*)shifts.data(), 0);
        dvz_segment_color(axes, 0, n_segs, col.data(), 0);
        dvz_segment_linewidth(axes, 0, n_segs, widths.data(), 0);
        dvz_panel_visual(ctx.panel, axes, 0);
    }
    return ctx;
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

BenchmarkResult runExperiment(PointsInput input)
{
    const size_t n         = input.pts.count;
    const size_t pos_bytes = sizeof(float) * 3 * n;

    // VRAM checkpoint #1: before we touch the GPU -- baseline held by other processes.
    const double vram_external = sampleVramUsedMB();

    checkCuda(cudaSetDevice(0));
    // VRAM checkpoint #2: force the primary CUDA context to exist (cudaFree(0)) and measure its
    // reservation on its own, before any of our buffers or the Vulkan renderer are created.
    checkCuda(cudaFree(0));
    double cuda_ctx_mb = sampleVramUsedMB() - vram_external;
    if (cuda_ctx_mb < 0.0) cuda_ctx_mb = 0.0;

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

    // Our sim device buffers (VRAM only; the pinned host buffer is host RAM): d_pos + rng + clusters.
    const double cuda_buf_mb =
        (double)(pos_bytes + rngStatesBytes(rng) + clusterBytes(clusters, n)) / (1024.0 * 1024.0);

    std::atomic<bool> quit_flag{false};

    DatovizContext ctx{};
    HudData hud{};
    hud.points  = (unsigned int)n;
    hud.seed    = input.pts.seed;
    hud.k       = input.pts.k;
    hud.epsilon = input.pts.epsilon;
    hud.fly     = input.fly;

    if (input.display)
    {
        ctx = setupDatoviz(input, h_pos, (unsigned int)n);
        dvz_app_gui(ctx.app, dvz_figure_id(ctx.figure), hudCallback, &hud);
        dvz_app_on_keyboard(ctx.app, keyCallback, &quit_flag);

        // GPU name/total-VRAM need NVML (filled after GPUPowerBegin below). The VRAM breakdown is
        // ready now: external + cuda_ctx are measured (checkpoints), buf + render are computed.
        hud.vram_external_mb = (float)vram_external;
        hud.cuda_ctx_mb      = (float)cuda_ctx_mb;
        hud.buf_mb           = (float)cuda_buf_mb;                        // device sim buffers
        hud.render_mb        = (float)(ctx.render_bytes / (1024.0 * 1024.0));
        // Vulkan render targets datoviz's canvas allocates (render_utils.h / canvas.c), no MSAA:
        // 3 swapchain + 1 staging color images (B8G8R8A8, 4 B) + 3 depth images (D32_SFLOAT, 4 B),
        // each width*height. Everything else on the graphics side is datoviz's own structures.
        const double px = (double)input.win_width * (double)input.win_height;
        hud.vulkan_mb = (float)(px * 4.0 * (4 /*color*/ + 3 /*depth*/) / (1024.0 * 1024.0));
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
    float total_pack     = 0.f;   // phong-mesh: per-frame center -> N*V vertex expansion (0 otherwise)
    float total_d2h      = 0.f;
    float total_h2h      = 0.f;
    float total_graphics = 0.f;
    size_t frame_count   = 0;
    int    iters_run     = 0;  // iterations the totals above accumulate over (totals / iters = per-frame avg)

    // phong-mesh reuses this buffer each frame for the expanded (N*V) vertex positions.
    std::vector<Vec3> mesh_expanded;
    if (input.display && input.mode == DvzMode::Mesh)
        mesh_expanded.resize(n * ctx.tmpl_verts.size());

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
        iters_run = i + 1;
        // --timeout: stop the run early (still falls through to finalize + CSV below).
        if (input.timeout_s > 0.f &&
            std::chrono::duration<double>(clk::now() - loop_start).count() >= input.timeout_s)
        { iters_run = i; break; }
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

            // --- Pack (phong-mesh only): expand N centers into N*V vertex positions on the CPU.
            // This is the price of datoviz having no per-instance streaming; it keeps D2H to just
            // the N centers (so only the unavoidable N*V H2D crosses PCIe, not a needless round
            // trip). Zero for the point-sprite modes, whose positions are already render-ready. ---
            float pack_ms = 0.f;
            float h2h_ms  = 0.f;
            if (input.mode == DvzMode::Mesh)
            {
                const uint32_t V = (uint32_t)ctx.tmpl_verts.size();
                auto t_pack = clk::now();
                expandMesh(ctx.tmpl_verts, (vec3*)h_pos, (uint32_t)n, input.size_px / 100.f, mesh_expanded.data());
                pack_ms = (float)ms_since(t_pack);

                // --- H2H: expanded verts -> datoviz's internal heap copy (N*V, not N) ---
                auto t_h2h = clk::now();
                dvz_mesh_position(ctx.visual, 0, (uint32_t)n * V, (vec3*)mesh_expanded.data(), 0);
                h2h_ms = (float)ms_since(t_h2h);
            }
            else
            {
                // --- H2H: pinned host -> datoviz's internal heap copy (CPU memcpy) ---
                auto t_h2h = clk::now();
                switch (input.mode)
                {
                case DvzMode::Sphere: dvz_sphere_position(ctx.visual, 0, (uint32_t)n, (vec3*)h_pos, 0); break;
                case DvzMode::Point:  dvz_point_position (ctx.visual, 0, (uint32_t)n, (vec3*)h_pos, 0); break;
                case DvzMode::Disc:   dvz_marker_position(ctx.visual, 0, (uint32_t)n, (vec3*)h_pos, 0); break;
                case DvzMode::Mesh:   break; // handled above
                }
                h2h_ms = (float)ms_since(t_h2h);
            }

            total_pack += pack_ms;
            total_d2h  += d2h_ms;
            total_h2h  += h2h_ms;

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

            float frame_total = compute_ms + pack_ms + d2h_ms + h2h_ms + graphics_ms;
            hud.frame      = (int)frame_count;
            hud.compute_ms = compute_ms;
            hud.pack_ms    = pack_ms;
            hud.d2h_ms     = d2h_ms;
            hud.h2h_ms     = h2h_ms;
            hud.render_ms  = graphics_ms;
            if (frame_total > 0.f) {
                float new_fps = 1000.f / frame_total;
                hud.fps = (frame_count == 1) ? new_fps : 0.9f * hud.fps + 0.1f * new_fps;
            }
            float watts   = (float)getGPUCurrentPower();
            hud.gpu_watts = (frame_count == 1) ? watts : 0.9f * hud.gpu_watts + 0.1f * watts;

            // Live VRAM-in-use (NVML), sampled every 30 frames to keep the query off the hot path.
            // Post-init this is stable; comparing it against buf_mb+render_mb tells us how much of
            // the used VRAM our own buffers account for (the rest is driver/Vulkan/other processes).
            if (frame_count == 1 || (frame_count % 30) == 0)
            {
                nvmlMemory_v2_t used_mi;
                used_mi.version = (unsigned int)(sizeof(used_mi) | (2 << 24U));
                if (nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &used_mi) == NVML_SUCCESS)
                    hud.vram_used_mb = (float)(used_mi.used / (1024.0 * 1024.0));
            }
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
    metrics.transfer.pack  = total_pack / 1000.f;   // phong-mesh vertex expansion (0 otherwise)
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

    return BenchmarkResult{ .perf = metrics, .power = gpu_power, .memory = nvml, .iters = iters_run };
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
        "  --geometry G       point        = raw point sprites (dvz_point), cheapest,\n"
        "                     sprite       = flat disc markers (dvz_marker),\n"
        "                     impostor     = lit sphere impostors (dvz_sphere),\n"
        "                     mesh         = lit triangle icospheres (dvz_mesh), mimir's\n"
        "                                    mesh geometry; one merged N-sphere mesh,\n"
        "                                    positions re-uploaded every frame (no instancing\n"
        "                                    in datoviz) -- costly, for comparison only\n"
        "                                                            (default: sprite)\n"
        "                     'impostor' matches benchmark_mimir --geometry impostor: the\n"
        "                     sphere radius is size/100 in [-1,1] domain units. Each datoviz\n"
        "                     primitive is lit or not by construction, so --shading only picks\n"
        "                     between them here; path-traced exits (raster baseline).\n"
        "                     Aliases: --shading, --render-path, --light-model, with the old\n"
        "                     value spellings (none/phong/phong-mesh/path-tracing).\n"
        "  --subdiv N         Icosphere subdivisions for mesh         (default: 2)\n"
        "                     0/1/2 = 20/80/320 triangles per sphere; matches mimir --subdiv.\n"
        "  --k N              Gaussian modes (clusters) at init     (default: 8)\n"
        "  --epsilon E        Per-axis stddev of each mode          (default: 0.05)\n"
        "                     The walk is mean-reverting, so clusters keep this\n"
        "                     stddev over time. Centers are seed-deterministic;\n"
        "                     same cloud as benchmark_mimir for equal args.\n"
        "  --background C     Panel background color: grey 'G' or 'R,G,B' in [0,1]\n"
        "                     (default: 0 = black)\n"
        "  --pcolor C         Particle color: grey 'G' or 'R,G,B' in [0,1]\n"
        "                     (default: 1 = white)\n"
        "  --fly              FPS fly camera (dvz_panel_fly, fixed up-vector) instead\n"
        "                     of the default arcball; matches benchmark_mimir --fly.\n"
        "  --axes             Draw the world +XYZ orientation triad at the origin\n"
        "  --timeout SECS     Stop after SECS wall-clock even if iters remain; the run still\n"
        "                     finalizes and writes its CSV row         (default: 0 = no timeout)\n"
        "                     (X=red, Y=green, Z=blue; letter labels at the tips) as an\n"
        "                     unlit depth-free overlay. Same triad as benchmark_mimir.\n"
        "\n"
        "(datoviz has no present-mode selection; the mimir --present flag has no\n"
        " datoviz equivalent, so it is intentionally absent here.)\n"
        "Rendering GPU: defaults to Vulkan device 0 to match the CUDA device; if your\n"
        "        CUDA and Vulkan device orders differ, set DVZ_GPU=<idx> to the Vulkan\n"
        "        index of the CUDA GPU.\n"
        "Keys: F1 toggles the HUD for clean screenshots; Ctrl+W quits.\n"
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
        if (a == "--fly")  { input.fly  = true; continue; } // valueless flags
        if (a == "--axes") { input.axes = true; continue; }
        if (a.rfind("--", 0) == 0)
        {
            if (i + 1 >= argc)
            { fprintf(stderr, "Missing value for %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
            std::string v = argv[++i];
            if      (a == "--vsync")       input.vsync    = (bool)std::stoi(v);
            else if (a == "--display")     input.display  = (bool)std::stoi(v);
            else if (a == "--size")        input.size_px  = std::stof(v);
            // --geometry is the axis that selects a datoviz primitive; --shading only says
            // whether it is lit, which follows from the primitive here, so both --shading and the
            // pre-split --render-path/--light-model spellings map through the same table.
            else if (a == "--geometry" || a == "--shading"
                  || a == "--render-path" || a == "--light-model")
                                           input.mode = parseGeometryMode(v);
            else if (a == "--subdiv")      input.subdiv = (uint32_t)std::stoul(v);
            else if (a == "--k")           input.pts.k = (unsigned int)std::stoul(v);
            else if (a == "--epsilon")     input.pts.epsilon = std::stof(v);
            else if (a == "--background")  input.background = parseColor(v);
            else if (a == "--pcolor")      input.pcolor = parseColor(v);
            else if (a == "--timeout")     input.timeout_s = std::stof(v);
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
