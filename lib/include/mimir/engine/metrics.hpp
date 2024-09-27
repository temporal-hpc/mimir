#pragma once

#include <vulkan/vulkan.h>

#include <chrono> // std::chrono
#include <vector> // std::vector

namespace mimir
{

struct MetricsCollector
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

    static MetricsCollector make(VkDevice device, uint32_t query_count, float period, size_t storage_size);
    double getRenderTimeResults(VkDevice device, uint32_t cmd_idx);
    float getFramerate();

    void startFrameWatch();
    float stopFrameWatch();

    void startRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
    void stopRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
};

static_assert(std::is_default_constructible_v<MetricsCollector>);
//static_assert(std::is_nothrow_default_constructible_v<MetricsCollector>);
//static_assert(std::is_trivially_default_constructible_v<MetricsCollector>);

} // namespace mimir