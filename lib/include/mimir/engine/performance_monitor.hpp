#include <cuda_runtime_api.h>

namespace mimir
{

struct PerformanceMonitor
{
    cudaStream_t stream;
    float total_compute_time;
    cudaEvent_t start, stop;

    static PerformanceMonitor make(cudaStream_t monitored_stream = 0)
    {
        PerformanceMonitor monitor{
            .stream = monitored_stream,
            .total_compute_time = 0,
            .start = nullptr,
            .stop = nullptr,
        };
        cudaEventCreate(&monitor.start);
        cudaEventCreate(&monitor.stop);
        return monitor;
    }
    void startCuda()
    {
        cudaEventRecord(start, stream);
    }
    float endCuda()
    {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float timems = 0;
        cudaEventElapsedTime(&timems, start, stop);
        total_compute_time += timems / 1000.f;
        return timems;
    }
};

static_assert(std::is_default_constructible_v<PerformanceMonitor>);
static_assert(std::is_nothrow_default_constructible_v<PerformanceMonitor>);
static_assert(std::is_trivially_default_constructible_v<PerformanceMonitor>);

} // namespace mimir