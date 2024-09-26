#include <mimir/engine/performance_monitor.hpp>

namespace mimir
{

PerformanceMonitor PerformanceMonitor::make(cudaStream_t monitored_stream)
{
    PerformanceMonitor monitor{
        .stream             = monitored_stream,
        .total_compute_time = 0,
        .start              = nullptr,
        .stop               = nullptr,
    };
    cudaEventCreate(&monitor.start);
    cudaEventCreate(&monitor.stop);
    return monitor;
}

void PerformanceMonitor::startCuda()
{
    cudaEventRecord(start, stream);
}

float PerformanceMonitor::endCuda()
{
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float timems = 0;
    cudaEventElapsedTime(&timems, start, stop);
    total_compute_time += timems / 1000.f;
    return timems;
}

} // namespace mimir