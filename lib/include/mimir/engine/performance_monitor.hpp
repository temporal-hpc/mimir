#include <cuda_runtime_api.h>

namespace mimir
{

struct PerformanceMonitor
{
    cudaStream_t stream = 0;
    float total_compute_time = 0;
    cudaEvent_t start = nullptr, stop = nullptr;

    static PerformanceMonitor make(cudaStream_t monitored_stream = 0)
    {
        PerformanceMonitor monitor{.stream = monitored_stream};
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

} // namespace mimir