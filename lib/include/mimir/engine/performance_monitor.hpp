#include <cuda_runtime_api.h>

#include <type_traits>

namespace mimir
{

struct PerformanceMonitor
{
    cudaStream_t stream;
    float total_compute_time;
    cudaEvent_t start, stop;

    static PerformanceMonitor make(cudaStream_t monitored_stream = 0);
    void startCuda();
    float endCuda();
};

static_assert(std::is_default_constructible_v<PerformanceMonitor>);
static_assert(std::is_nothrow_default_constructible_v<PerformanceMonitor>);
static_assert(std::is_trivially_default_constructible_v<PerformanceMonitor>);

} // namespace mimir