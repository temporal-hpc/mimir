#pragma once

// Standalone benchmark input for the datoviz N-body comparison sample.
// Mirrors samples/nbody/benchmark.hpp (same positional args and defaults) so the two
// binaries are drop-in comparable, but without any mimir dependency. datoviz has no
// present-mode selection, so there is no `present` field here; `vsync` is real display
// vsync (DVZ_CANVAS_FLAGS_VSYNC), unlike mimir's --interop-sync.

struct BenchmarkInput
{
    int width;
    int height;
    unsigned int body_count;
    int iter_count;
    bool vsync;    // real display vsync (DVZ_CANVAS_FLAGS_VSYNC)
    bool display;
    bool use_cpu;

    // Default experiment parameters (same as samples/nbody).
    static BenchmarkInput defaultValues()
    {
        return BenchmarkInput{
            .width       = 1920,
            .height      = 1080,
            .body_count  = 77824,
            .iter_count  = 1000000,
            .vsync       = true,
            .display     = true,
            .use_cpu     = false,
        };
    }
};
