// Headless verification for LightModel::PathTracing. Fills the interop position buffer with a
// deterministic grid, renders the scene (per-frame TLAS built from those positions) to a PPM.
// Not part of the benchmark; a throwaway acceptance check for Phase 2's dynamic scene path.
#include <mimir/mimir.hpp>

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mimir;

int main(int argc, char** argv)
{
    ViewerOptions opts;
    opts.render_mode      = RenderMode::Headless;
    opts.light_model      = LightModel::PathTracing;
    opts.window.size      = { 512, 512 };
    opts.background_color  = { 0.10f, 0.10f, 0.13f, 1.f }; // dark grey: exercises env fill light
    opts.light_pos         = { -0.4082f, 0.4082f, -0.8165f }; // match benchmark PT world sun
    opts.pt_samples_per_pixel = 16; // more samples to expose GI/shadow noise cleanly
    opts.pt_subdivisions      = 2;  // exercise BLAS tessellation (320 tris)
    opts.pt_max_bounces       = (argc > 2) ? (unsigned)atoi(argv[2]) : 4; // GI depth (argv[2])
    opts.present.enable_fps_limit = false;

    InstanceHandle engine = nullptr;
    createInstance(opts, &engine);

    // A 5x5x5 grid of particles in [-1,1]^3, driving the dynamic TLAS.
    constexpr int N = 5;
    constexpr unsigned int n = N * N * N;
    float* d_pos = nullptr;
    AllocHandle pos_alloc{};
    allocLinear(engine, (void**)&d_pos, sizeof(float) * 3 * n, &pos_alloc);

    std::vector<float> host(3 * n);
    unsigned int idx = 0;
    for (int z = 0; z < N; ++z)
    for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x)
    {
        auto coord = [](int i){ return -1.f + 2.f * (float(i) + 0.5f) / float(N); };
        host[3 * idx + 0] = coord(x);
        host[3 * idx + 1] = coord(y);
        host[3 * idx + 2] = coord(z);
        idx++;
    }
    cudaMemcpy(d_pos, host.data(), sizeof(float) * 3 * n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    ViewDescription desc{
        .type       = ViewType::Markers,
        .domain     = DomainType::Domain3D,
        .attributes = {
            { AttributeType::Position, AttributeDescription{
                .source = pos_alloc,
                .size   = n,
                .format = FormatDescription::make<float3>(),
            }}
        },
        .layout        = Layout::make(n),
        .default_color = { 0.90f, 0.50f, 0.20f, 1.f }, // warm orange: verifies --pcolor -> PT albedo
        .default_size  = 0.12f, // world radius of each icosphere instance
    };
    ViewHandle view = nullptr;
    createView(engine, &desc, &view);

    setCameraPosition(engine, { 0.f, 0.f, -4.f });
    renderHeadless(engine, []{}, 3);

    const char* path = argc > 1 ? argv[1] : "pt_check.ppm";
    saveFrame(engine, path);
    printf("saved %s\n", path);

    destroyInstance(engine);
    return 0;
}
