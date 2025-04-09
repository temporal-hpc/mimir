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
    // Window title when shown
    std::string title = "Mimir";
    // Window size in (width, height) format.
    int2 size         = { 800, 600 };

    static WindowOptions makeDefault()
    {
        return WindowOptions{ .title = "Mimir", .size = { 800, 600 } };
    }
};

enum class PresentMode { Immediate, TripleBuffering, VSync };

struct PresentOptions
{
    // Sets frame presentation scheme used by the engine instance.
    PresentMode mode;
    // Enable/disable CUDA-Vulkan interop synchronization.
    bool enable_sync;
    // Enables the FPS cap with the value specified by 'target_fps'.
    bool enable_fps_limit;
    // Throttle rendering to achieve this value when 'enable_fps_limit' is enabled.
    int target_fps;
    int64_t target_frame_time;

    static PresentOptions makeDefault()
    {
        return PresentOptions{
            .mode              = PresentMode::Immediate,
            .enable_sync       = true,
            .enable_fps_limit  = true,
            .target_fps        = 60,
            .target_frame_time = 0,
        };
    }
};

struct ViewerOptions
{
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
};

} // namespace mimir

