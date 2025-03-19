#pragma once

#include <cuda_runtime_api.h>

#include <string> // std::string

namespace mimir
{

struct PerformanceMetrics
{
    float frame_rate;
    struct {
        float compute;
        float graphics;
        float pipeline;
    } times;

    struct {
        float usage;
        float budget;
    } devmem;
};

struct WindowOptions
{
    std::string title = "Mimir";
    int2 size         = { 800, 600 };
    bool fullscreen   = false; // TODO: Implement
};

enum class PresentMode { Immediate, TripleBuffering, VSync };

struct PresentOptions
{
    PresentMode mode        = PresentMode::Immediate;
    bool enable_sync        = true;
    bool enable_fps_limit   = false;
    int target_fps          = 60;
    int max_fps             = 0;
    float target_frame_time = 0.f;
};

struct ViewerOptions
{
    WindowOptions window    = {};
    PresentOptions present  = {};
    bool show_panel         = false;
    bool show_metrics       = false;
    bool show_demo_window   = false;
    float4 background_color = {.5f, .5f, .5f, 1.f};
};

} // namespace mimir