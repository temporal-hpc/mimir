#include <mimir/engine/metrics.hpp>

#include "internal/validation.hpp"

namespace mimir
{

VkQueryPool createQueryPool(VkDevice device, uint32_t query_count)
{
    // Number of queries is twice the number of command buffers, to store space
    // for queries before and after rendering
    VkQueryPoolCreateInfo info{
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext      = nullptr,
        .flags      = 0,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = query_count, //command_buffers.size() * 2;
        .pipelineStatistics = 0,
    };

    VkQueryPool pool = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateQueryPool(device, &info, nullptr, &pool));
    vkResetQueryPool(device, pool, 0, query_count);
    return pool;
}

MetricsCollector MetricsCollector::make(VkDevice device, uint32_t query_count, float period, size_t storage_size)
{
    return {
        .query_pool          = createQueryPool(device, query_count),
        .frame_times         = std::vector<float>(storage_size, 0.f),
        .total_pipeline_time = 0.0,
        .total_graphics_time = 0.f,
        .total_frame_count   = 0,
        .timestamp_period    = period,
    };
}

void MetricsCollector::advanceFrame(float frame_time)
{
    total_graphics_time += frame_time;
    frame_times[total_frame_count % frame_times.size()] = frame_time;
    total_frame_count++;
}

double MetricsCollector::getRenderTimeResults(VkDevice device, uint32_t cmd_idx)
{
    uint64_t buffer[2];
    validation::checkVulkan(vkGetQueryPoolResults(device, query_pool,
        2 * cmd_idx, 2, 2 * sizeof(uint64_t), buffer, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
    );
    vkResetQueryPool(device, query_pool, cmd_idx * 2, 2);
    // TODO: apply time &= timestamp_mask;
    auto seconds_per_tick = static_cast<double>(timestamp_period) / 1e9;
    return static_cast<double>(buffer[1] - buffer[0]) * seconds_per_tick;
}

float MetricsCollector::getFramerate()
{
    auto frame_sample_size = std::min(frame_times.size(), total_frame_count);
    float total_frame_time = 0;
    for (size_t i = 0; i < frame_sample_size; ++i) total_frame_time += frame_times[i];
    return frame_times.size() / total_frame_time;
}

} // namespace mimir