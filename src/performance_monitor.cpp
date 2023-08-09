#include <cudaview/engine/performance_monitor.hpp>

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
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);    
    total_compute_time += elapsed;
    return elapsed;
}