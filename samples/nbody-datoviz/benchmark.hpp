#pragma once

// Standalone benchmark input for the datoviz N-body comparison sample.
// Mirrors samples/nbody/benchmark.hpp field-for-field (and CLI argument order) so the
// two binaries are drop-in comparable, but without any mimir dependency. `present` is
// kept as a plain int to preserve the CLI argument layout; datoviz does not expose
// mimir's PresentMode enum.

struct BenchmarkInput
{
    int width;
    int height;
    unsigned int body_count;
    int iter_count;
    int present;       // kept for CLI parity with nbody (unused by datoviz path)
    int target_fps;
    bool enable_sync;  // maps to vsync when displaying
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
            .present     = 0,
            .target_fps  = 0,
            .enable_sync = true,
            .display     = true,
            .use_cpu     = false,
        };
    }
};
