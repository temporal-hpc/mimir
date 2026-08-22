#pragma once

#include <cuda_runtime_api.h>

#include <string> // std::string

namespace mimir
{

// Container for performance data collected by the engine while running
struct PerformanceMetrics
{
    float frame_rate;
    struct {
        // Total time used by CUDA kernel calls.
        float compute;
        // Total time taken by the engine between frame acquisition and presentation.
        float graphics;
        // Total time taken by the shader pipeline during rendering.
        float pipeline;
        // Path tracing only (0 otherwise): last-frame GPU time to rebuild the TLAS from the live
        // positions (instance writer + build), and to trace the frame (vkCmdTraceRays), in ms.
        float tlas_build;
        float trace;
        // Last-frame CPU-side phase breakdown of the render thread (ms). Unlike `pipeline` (the
        // GPU render-pass timestamp), these cover the parts of a frame the GPU timestamp misses:
        //   wait   = blocked on the frame fence + swapchain acquire (GPU / present backpressure)
        //   record = CPU command-buffer recording
        //   submit = vkQueueSubmit + vkQueuePresentKHR
        float wait;
        float record;
        float submit;
        // True end-to-end GPU frame latency (submit -> fence signalled), lockstep interop only.
        // The honest measure of a frame's GPU cost when the render-pass timestamp under-reports.
        float gpu;
    } times;

    struct {
        // GPU memory usage as measured by the Vulkan API.
        float usage;
        // GPU available memory as measured by the Vulkan API.
        float budget;
    } devmem;
};

// Window creation parameters
struct WindowOptions
{
    // Window title when shown.
    std::string title = "Mimir";
    // Window size in (width, height) format.
    int2 size     = { 800, 600 };
    // Whether to show/hide the upper window panel.
    bool decorate = true;
    // Whether to show/hide the display window.
    bool visible  = true;

    static WindowOptions makeDefault()
    {
        return WindowOptions{
            .title = "Mimir",
            .size = { 800, 600 },
            .decorate = true,
            .visible = true
        };
    }
};

// Selects where rendered frames are sent.
// Local    = on-screen window (default behavior).
// Headless = offscreen render target, no window or surface (foundation for remote rendering).
// Remote   = headless rendering streamed to a connected client (not yet implemented).
enum class RenderMode { Local, Headless, Remote };

// The two orthogonal axes of how elements are drawn. They were a single conflated enum
// (LightModel, then RenderPath) until 2026-08: Impostor and Mesh are the SAME Blinn-Phong lighting
// on different geometry, and the old `Flat` welded a geometry choice (2D sprite) to a lighting one
// (unlit), so neither question could be answered without answering the other.

// How surfaces are lit -- increasing fidelity, and nothing about their shape.
// Unlit      = no lighting; elements are drawn in their own color.
// Phong      = Blinn-Phong local lighting from the scene light (the historical default look).
// PathTraced = Vulkan ray-traced path tracing: shadows and global illumination (pathtrace.slang).
//              Requires an RT-capable GPU; falls back to Phong raster with a warning otherwise.
//              The tracer's primitives are analytic AABB spheres, so Geometry does not apply.
enum class Shading { Unlit, Phong, PathTraced };

// What each marker IS -- its primitive shape, and nothing about how it is lit. Applies to
// ViewType::Markers; other view types have an inherent shape (a voxel is a cube) and ignore it.
// Sprite   = flat 2D point-sprite disc, screen-space and sizeless in depth (marker_flat.slang).
//            Cheapest, and the datoviz-comparable point/disc primitive. Has no surface normals,
//            so it renders unlit whatever the Shading says.
// Impostor = one screen-aligned quad per marker, with the sphere recovered analytically by a
//            ray-sphere intersection in the fragment shader, including a true depth write
//            (marker.slang). Perfectly round at any zoom.
// Mesh     = instanced triangle icosphere per marker (marker_mesh.slang), tessellation from
//            ViewerOptions::mesh_subdivisions. No geometry stage and no per-fragment ray-sphere,
//            so early-Z culls overdraw -- cheaper than Impostor at high resolution.
// The ordinals of both enums are part of the remote protocol -- do not reorder.
enum class Geometry { Sprite, Impostor, Mesh };

enum class PresentMode { Immediate, TripleBuffering, VSync };

// On-screen camera interaction. Orbit keeps the historical drag controls (left-drag = rotate,
// right-drag = zoom, middle-drag = pan). Fly is a captured-mouse-look FPS camera: the cursor is
// locked and WASD flies (Q/E or Space/LCtrl for down/up), with a key to release the cursor for
// the ImGui HUD. Ignored in headless/scripted auto-orbit runs.
enum class CameraControl { Orbit, Fly };

// Keys reportable through the input API (isKeyDown/isKeyPressed). Mimir's own enum, so samples do
// keyboard input without depending on GLFW or ImGui. `Count` bounds the internal state arrays.
enum class Key : int {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Left, Right, Up, Down,
    Space, Enter, Escape, Tab, Comma, Period,
    Count
};

struct PresentOptions
{
    // Sets frame presentation scheme used by the engine instance.
    PresentMode mode;
    // Enable/disable CUDA-Vulkan interop synchronization. Note this is NOT display vsync
    // (that is selected via 'mode' == PresentMode::VSync); it gates compute/render access to
    // the shared interop buffer via the timeline-semaphore handshake.
    bool enable_interop_sync;
    // Enables the FPS cap with the value specified by 'target_fps'.
    bool enable_fps_limit;
    // Throttle rendering to achieve this value when 'enable_fps_limit' is enabled.
    int target_fps;
    int64_t target_frame_time;

    static PresentOptions makeDefault()
    {
        return PresentOptions{
            .mode                = PresentMode::Immediate,
            .enable_interop_sync = true,
            .enable_fps_limit    = true,
            .target_fps        = 60,
            .target_frame_time = 0,
        };
    }
};

struct ViewerOptions
{
    // Selects on-screen (Local) vs offscreen/headless rendering for this instance.
    RenderMode render_mode  = RenderMode::Local;

    // Instance-wide shading and geometry -- the two independent axes above. Together they decide
    // which pipeline each Markers view is built with; Phong + Impostor preserves the historical
    // default look. Combinations a renderer cannot honour (a lit Sprite, any Geometry under
    // PathTraced) warn once at createView and render the nearest sensible thing.
    Shading  shading  = Shading::Phong;
    Geometry geometry = Geometry::Impostor;

    // Forces mimir's CUDA-Vulkan interop device to this CUDA device ordinal (from the
    // CUDA-visible set), instead of auto-picking the first suitable GPU. Set this to match
    // whatever device your own CUDA/compute code runs on (e.g. a --dev CLI argument) on a
    // multi-GPU host -- CUDA device pointers aren't valid across GPUs without explicit
    // peer-to-peer access, so interop buffers must live on the same physical GPU as your
    // compute. -1 (default) = auto-pick.
    int cuda_device = -1;

    // Options for the window associated to the engine instance.
    WindowOptions window    = WindowOptions::makeDefault();

    // Frame presentation options associated to the engine instance.
    PresentOptions present  = PresentOptions::makeDefault();

    // Master GUI switch: when false, NO ImGui windows are drawn (engine panel, metrics/demo
    // windows, and the setGuiCallback overlay), leaving a clean viewport for screenshots.
    // Toggled at runtime with F1.
    bool show_gui           = true;

    // Show/hide the control panel for camera/scene/view data.
    bool show_panel         = false;

    // Show/hide the ImGUI metrics panel.
    bool show_metrics       = false;

    // Show/hide the ImGUI demo window.
    bool show_demo_window   = false;

    // Built-in performance overlay: a small always-on-top corner readout of FPS, frame time and
    // GPU render-pass time, drawn by the engine itself. Lets interactive samples show a HUD without
    // depending on ImGui or writing any GUI code. Toggled at runtime with F2; hidden with F1 (the
    // master show_gui switch) like every other window.
    bool show_hud           = false;

    // Background color for the current engine instance.
    float4 background_color = {.5f, .5f, .5f, 1.f};

    float3 light_pos = { 0.f, 0.f, -1.f };
    // Light color/intensity, applied across ALL light models: the raster Phong modes use it as
    // the sun's diffuse/specular intensity, and path tracing scales its sun radiance by it
    // (radiance = 6 * light_color, so this 0.5 default keeps PT's historical 3.0).
    float3 light_color = { .5f, .5f, .5f };
    float3 specular_color = { 1.f, 1.f, 1.f };
    float specular_power = 32.f;
    float ambient_strength = .05f;

    // Path-tracing workload knobs (only used when shading == Shading::PathTraced;
    // ignored by the raster light models). See DESIGN_pathtracing.md §6.
    unsigned int pt_samples_per_pixel = 1; // rays per pixel per frame (--spp)
    unsigned int pt_max_bounces       = 4; // max path depth (--bounces)
    unsigned int mesh_subdivisions    = 1; // icosphere tessellation: 0=20,1=80,2=320 tris (--subdiv)
    // BLAS refit cadence: a full rebuild happens every pt_rebuild_interval dirty frames, with cheap
    // in-place refits in between (see RayTracingContext::rebuild_interval in raytracing.hpp). <= 1
    // disables refit (full rebuild every frame). Larger trades traversal quality for speed as the
    // scene deforms between rebuilds. (--bvh-refit)
    unsigned int pt_rebuild_interval  = 8;
    // Denoise the path-traced result before display (--denoise) with a Vulkan-compute a-trous
    // edge-avoiding wavelet filter (runs on any GPU). Edge-stopping is guided by the first-hit
    // normal/depth G-buffer, so noise is smoothed while silhouettes and shading edges survive.
    bool pt_denoise = false;
    // Level-of-detail data reduction: N = cells per axis of an N^3 voxel grid over the [-1,1]^3
    // domain. 0 (default) = one primitive per particle (no LOD). N>0 draws one representative per
    // occupied cell (at the cell's mass centroid), trading fidelity for build/trace/draw speed.
    // NOTE: despite sitting among the path-tracing knobs, this is TRANSVERSAL -- it applies to ALL
    // light models (none/phong/phong-mesh/path-tracing), reducing how many primitives each renderer
    // draws. The caller must bound N to available VRAM (the N^3 accumulator is up to 32 B/cell) and to
    // N <= 1625 (cells are indexed in uint32, so N^3 must stay < 2^32).
    // Execution: the reduction (clear/scatter/emit) runs on custom native-CUDA kernels by default --
    // single-digit ms at hundreds of millions of particles, and (unlike a CUB-based reduction) not
    // capped at a 32-bit item count, so it scales past 2^32 particles. Set MIMIR_LOD_NO_CUDA=1 to force
    // the Vulkan-compute scatter/emit fallback instead (also used automatically when CUDA/Vulkan
    // interop is unavailable); it produces identical occupied-cell counts/positions, just slower,
    // especially under the centroid placement's atomic contention (see lod_centroid below).
    //
    // ViewType::Voxels reinterprets this as a grid-COARSENING factor M: a fine N^3 int-state voxel grid
    // is max-pooled ON THE FLY in the vertex shader into an M^3 grid, and M^3 cubes are drawn (M < N;
    // M >= N or M == 0 is a no-op that draws the fine grid). No coarse data is materialized -- each
    // coarse cell reads its disjoint (N/M)^3 fine block directly and takes the max (state 0 = dead), so
    // it reuses the same sim->draw synchronization as the normal voxel color attribute and needs no
    // extra buffers/compute/interop sync. Requires the color attribute to be an int32 index (the
    // colormap-index type) over a true cubic N^3 grid; other cases draw the fine grid unchanged. Unlike
    // the Markers reduction above, this path needs no ray-tracing/BDA support. Other view types
    // (Image/Edges) ignore pt_lod_cells (a warning is logged).
    unsigned int pt_lod_cells = 0;

    // LOD representative placement (only meaningful when pt_lod_cells > 0):
    //   false (default) = each cell's representative sits at the cell's geometric CENTER. Skips the
    //                     centroid's 3 int64 position-sum atomics per particle, so the reduction is
    //                     markedly faster -- the atomics are the reduction bottleneck at huge N.
    //   true            = the mass CENTROID of the particles in the cell. Needs int64 atomics (auto-
    //                     falls back to cell-center without them); slightly more accurate positions.
    // Default is cell-center because the atomic cost dominates at scale: measured on an RTX PRO 6000
    // Blackwell at 300M particles/lod 256^3, centroid costs ~3x the reduction time of cell-center
    // (~20 ms vs ~6 ms), and the gap widens super-linearly with N (~11x at 6e9 due to distributed-L2
    // atomic contention). Opt into centroid (rr-server --lod-placement centroid) only when the finer
    // representative position matters more than reduction throughput.
    bool lod_centroid = false;

    // Render LOD representatives as solid grid-aligned cubes (voxels) instead of spheres. Default:
    // true. Applies only under pt_lod_cells > 0 and only to lit models (phong / phong-mesh /
    // path-tracing); `none` (flat points) ignores it. Forces cell-center placement and full-cell fill.
    // Set false (rr-server --lod-shape sphere) to render LOD as spheres and honour lod_centroid.
    // Independent of the CA3D voxel_boxes pipeline.
    bool lod_voxel = true;

    // Vertical field of view of the perspective camera, in degrees. Datoviz-comparable
    // samples set 45 to match datoviz's fixed GLM_PI_4 perspective, so both libraries
    // frame the same domain identically.
    float camera_fov = 40.f;

    // Camera interaction (on-screen only; see CameraControl).
    CameraControl camera_control = CameraControl::Orbit;
    float mouse_sensitivity   = 0.1f; // degrees of yaw/pitch per pixel of mouse motion (Fly)
    float camera_move_speed   = 3.f;  // world units/second for WASD movement (Fly)
    // Scripted auto-orbit for reproducible, input-free runs (e.g. benchmarks): the camera circles
    // the scene origin at this angular speed in degrees/second. >0 overrides manual control; 0 off.
    float orbit_speed         = 0.f;
};

} // namespace mimir

