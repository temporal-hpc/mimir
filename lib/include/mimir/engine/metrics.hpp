#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <chrono> // std::chrono
#include <vector> // std::vector

namespace mimir::metrics
{

struct ComputeMonitor
{
    cudaStream_t stream;
    float total_compute_time;
    cudaEvent_t start, stop;

    static ComputeMonitor make(cudaStream_t monitored_stream = 0);
    void startWatch();
    float stopWatch();
};

static_assert(std::is_default_constructible_v<ComputeMonitor>);
static_assert(std::is_nothrow_default_constructible_v<ComputeMonitor>);
static_assert(std::is_trivially_default_constructible_v<ComputeMonitor>);

struct GraphicsMonitor
{
    VkQueryPool query_pool;
    std::vector<float> frame_times;
    double total_pipeline_time;
    float total_graphics_time;
    size_t total_frame_count;
    float timestamp_period;

    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TimePoint frame_start;
    TimePoint frame_end;

    static GraphicsMonitor make(VkDevice device, uint32_t query_count, float period, size_t storage_size);
    double getRenderTimeResults(VkDevice device, uint32_t cmd_idx);
    float getFramerate();

    void startFrameWatch();
    float stopFrameWatch();

    void startRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
    void stopRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
};

static_assert(std::is_default_constructible_v<GraphicsMonitor>);
//static_assert(std::is_nothrow_default_constructible_v<GraphicsMonitor>);
//static_assert(std::is_trivially_default_constructible_v<GraphicsMonitor>);

} // namespace mimir