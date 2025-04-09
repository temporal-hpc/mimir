#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

#include <chrono> // std::chrono
#include <vector> // std::vector

namespace mimir::metrics
{

struct ComputeMonitor
{
    // Cuda stream being monitored.
    cudaStream_t stream;
    // Accumulated time from all start/stop pairs of events.
    float total_compute_time;
    // Device handles for start/stop points of measured event.
    cudaEvent_t start, stop;

    // Creates a monitor instance for the given stream.
    static ComputeMonitor make(cudaStream_t monitored_stream = 0);
    // Starts compute stopwatch.
    void startWatch();
    // Stops compute stopwatch and adds measured time to the total.
    float stopWatch();
};

static_assert(std::is_default_constructible_v<ComputeMonitor>);
static_assert(std::is_nothrow_default_constructible_v<ComputeMonitor>);
static_assert(std::is_trivially_default_constructible_v<ComputeMonitor>);

struct GraphicsMonitor
{
    // Structure for creating graphics queue measurement points.
    VkQueryPool query_pool;
    // Moving window of the last registered frame times.
    std::vector<float> frame_times;
    // Accumulated time from all start/stop timestamp query events.
    double total_pipeline_time;
    // Accumulated time from all frame times measured from host.
    float total_graphics_time;
    // Total number of registered frames.
    size_t total_frame_count;
    // Number of nanoseconds per timestamp tick, used to get measure in time units.
    float timestamp_period;

    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    // Start/stop points in host for measuring frame time.
    TimePoint frame_start, frame_end;

    // Creates a graphics monitor with storage for a fixed amount of stored frame times.
    static GraphicsMonitor make(VkDevice device, uint32_t query_count, float period, size_t storage_size);

    double getRenderTimeResults(VkDevice device, uint32_t cmd_idx);
    // Gets average framerate using frame times stored in the moving window.
    float getFramerate();

    // Starts frame time stopwatch.
    void startFrameWatch();
    // Stops frame time stopwatch.
    float stopFrameWatch();

    // Starts graphics pipeline stopwatch.
    void startRenderWatch(VkDevice device, VkCommandBuffer cmd, uint32_t frame_idx);
    // Stops graphics pipeline stopwatch.
    void stopRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
};

static_assert(std::is_default_constructible_v<GraphicsMonitor>);
//static_assert(std::is_nothrow_default_constructible_v<GraphicsMonitor>);
//static_assert(std::is_trivially_default_constructible_v<GraphicsMonitor>);

} // namespace mimir