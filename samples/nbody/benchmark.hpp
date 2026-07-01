#include <mimir/mimir.hpp>

struct BenchmarkInput
{
    int width;
    int height;
    unsigned int body_count;
    int iter_count;
    mimir::PresentMode present;
    bool enable_interop_sync;
    bool display;
    bool use_cpu;

    // Default experiment parameters
    static BenchmarkInput defaultValues()
    {
        return BenchmarkInput{
            .width       = 1920,
            .height      = 1080,
            .body_count  = 77824,
            .iter_count  = 1000000,
            .present     = mimir::PresentMode::Immediate,
            .enable_interop_sync = true,
            .display     = true,
            .use_cpu     = false,
        };
    }
};