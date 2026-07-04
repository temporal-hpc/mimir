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

// Selects how the scene is shaded, instance-wide.
// None        = unlit raster; markers draw as flat 2D point-sprite discs.
// Phong       = lit raster; markers draw as ray-sphere impostors with Blinn-Phong (the
//               datoviz-comparable lit-sphere technique).
// PhongMesh   = lit raster; markers draw as instanced triangle icospheres (same geometry as
//               PathTracing, tessellation from pt_subdivisions). Cheaper than impostors at high
//               resolution (early-Z, no per-fragment ray-sphere) and geometry-matched to path
//               tracing. mimir-only: datoviz has no per-instance mesh path (keep it on Phong).
// PathTracing = Vulkan ray-traced path tracing (requires an RT-capable GPU; markers
//               become instanced triangle icospheres). In development — currently
//               falls back to Phong raster with a warning. See DESIGN_pathtracing.md.
enum class LightModel { None, Phong, PhongMesh, PathTracing };

enum class PresentMode { Immediate, TripleBuffering, VSync };

// On-screen camera interaction. Orbit keeps the historical drag controls (left-drag = rotate,
// right-drag = zoom, middle-drag = pan). Fly is a captured-mouse-look FPS camera: the cursor is
// locked and WASD flies (Q/E or Space/LCtrl for down/up), with a key to release the cursor for
// the ImGui HUD. Ignored in headless/scripted auto-orbit runs.
enum class CameraControl { Orbit, Fly };

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

    // Instance-wide shading model; drives how each view's pipeline is built
    // (e.g. flat vs lit markers). Phong preserves the historical default look.
    LightModel light_model  = LightModel::Phong;

    // Options for the window associated to the engine instance.
    WindowOptions window    = WindowOptions::makeDefault();

    // Frame presentation options associated to the engine instance.
    PresentOptions present  = PresentOptions::makeDefault();

    // Show/hide the control panel for camera/scene/view data.
    bool show_panel         = false;

    // Show/hide the ImGUI metrics panel.
    bool show_metrics       = false;

    // Show/hide the ImGUI demo window.
    bool show_demo_window   = false;

    // Background color for the current engine instance.
    float4 background_color = {.5f, .5f, .5f, 1.f};

    float3 light_pos = { 0.f, 0.f, -1.f };
    float3 light_color = { .5f, .5f, .5f };
    float3 specular_color = { 1.f, 1.f, 1.f };
    float specular_power = 32.f;
    float ambient_strength = .05f;

    // Path-tracing workload knobs (only used when light_model == LightModel::PathTracing;
    // ignored by the raster light models). See DESIGN_pathtracing.md §6.
    unsigned int pt_samples_per_pixel = 1; // rays per pixel per frame (--spp)
    unsigned int pt_max_bounces       = 4; // max path depth (--bounces)
    unsigned int pt_subdivisions      = 1; // icosphere tessellation: 0=20,1=80,2=320 tris (--subdiv)
    // Denoise the path-traced result before display (--denoise) with a Vulkan-compute a-trous
    // edge-avoiding wavelet filter (runs on any GPU). Edge-stopping is guided by the first-hit
    // normal/depth G-buffer, so noise is smoothed while silhouettes and shading edges survive.
    bool pt_denoise = false;

    // Camera interaction (on-screen only; see CameraControl).
    CameraControl camera_control = CameraControl::Orbit;
    float mouse_sensitivity   = 0.1f; // degrees of yaw/pitch per pixel of mouse motion (Fly)
    float camera_move_speed   = 3.f;  // world units/second for WASD movement (Fly)
    // Scripted auto-orbit for reproducible, input-free runs (e.g. benchmarks): the camera circles
    // the scene origin at this angular speed in degrees/second. >0 overrides manual control; 0 off.
    float orbit_speed         = 0.f;
};

} // namespace mimir

