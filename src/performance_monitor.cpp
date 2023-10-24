#include <mimir/engine/performance_monitor.hpp>

namespace mimir
{

PerformanceMonitor::PerformanceMonitor(cudaStream_t stream)
    : monitored_stream(stream)
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

void PerformanceMonitor::startCuda()
{
    cudaEventRecord(start, monitored_stream);
}

float PerformanceMonitor::endCuda()
{
    cudaEventRecord(stop, monitored_stream);
    cudaEventSynchronize(stop);
    float timems = 0;
    cudaEventElapsedTime(&timems, start, stop);    
    total_compute_time += timems / 1000.f;
    return timems;
}

} // namespace mimir