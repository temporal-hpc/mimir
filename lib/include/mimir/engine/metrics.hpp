#pragma once

#include <vulkan/vulkan.h>

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

    static MetricsCollector make(VkDevice device, uint32_t query_count, float period, size_t storage_size);
    void advanceFrame(float frame_time);
    double getRenderTimeResults(VkDevice device, uint32_t cmd_idx);
    float getFramerate();

    void startRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
    void stopRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx);
};

} // namespace mimir