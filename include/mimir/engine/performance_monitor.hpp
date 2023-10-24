#include <cuda_runtime_api.h>

namespace mimir
{

struct PerformanceMonitor
{
    PerformanceMonitor(cudaStream_t stream = 0);
    void startCuda();
    float endCuda();

    cudaStream_t monitored_stream = 0;
    float total_compute_time = 0;
    cudaEvent_t start = nullptr, stop = nullptr;
};

} // namespace mimir