#include "mimir/metrics.hpp"

#include "mimir/validation.hpp"

namespace mimir::metrics
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

GraphicsMonitor GraphicsMonitor::make(VkDevice device, uint32_t query_count, float period, size_t storage_size)
{
    return {
        .query_pool          = createQueryPool(device, query_count),
        .frame_times         = std::vector<float>(storage_size, 0.f),
        .total_pipeline_time = 0.0,
        .total_graphics_time = 0.f,
        .total_frame_count   = 0,
        .timestamp_period    = period,
        .frame_start         = {},
        .frame_end           = {}
    };
}

double GraphicsMonitor::getRenderTimeResults(VkDevice device, uint32_t cmd_idx)
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

float GraphicsMonitor::getFramerate()
{
    auto frame_sample_size = std::min(frame_times.size(), total_frame_count);
    float total_frame_time = 0;
    for (size_t i = 0; i < frame_sample_size; ++i) total_frame_time += frame_times[i];
    return frame_times.size() / total_frame_time;
}

void GraphicsMonitor::startFrameWatch()
{
    // If a frame has already been measured, use the previous end point as start for the next
    // Otherwise, use current time as start
    frame_start = (frame_end == TimePoint{})? std::chrono::high_resolution_clock::now() : frame_end;
}

float GraphicsMonitor::stopFrameWatch()
{
    namespace time = std::chrono;
    frame_end = time::high_resolution_clock::now();
    auto frame_time = time::duration<float,time::seconds::period>(frame_end - frame_start).count();

    total_graphics_time += frame_time;
    frame_times[total_frame_count % frame_times.size()] = frame_time;
    total_frame_count++;
    return frame_time;
}

void GraphicsMonitor::startRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx)
{
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, frame_idx * 2);
}

void GraphicsMonitor::stopRenderWatch(VkCommandBuffer cmd, uint32_t frame_idx)
{
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, frame_idx * 2 + 1);
    // if (render_timeline > MAX_FRAMES_IN_FLIGHT)
    // {
    //     total_pipeline_time += getRenderTimeResults(frame_idx);
    // }
}

ComputeMonitor ComputeMonitor::make(cudaStream_t monitored_stream)
{
    ComputeMonitor monitor{
        .stream             = monitored_stream,
        .total_compute_time = 0,
        .start              = nullptr,
        .stop               = nullptr,
    };
    cudaEventCreate(&monitor.start);
    cudaEventCreate(&monitor.stop);
    return monitor;
}

void ComputeMonitor::startWatch()
{
    cudaEventRecord(start, stream);
}

float ComputeMonitor::stopWatch()
{
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float timems = 0;
    cudaEventElapsedTime(&timems, start, stop);
    total_compute_time += timems / 1000.f;
    return timems;
}

} // namespace mimir