// Headless verification for RenderPath::PathTraced. Fills the interop position buffer with a
// deterministic grid, renders the scene (per-frame TLAS built from those positions) to a PPM.
// Not part of the benchmark; a throwaway acceptance check for Phase 2's dynamic scene path.
//
// PT_COLORS=1 additionally binds a per-primitive Color attribute (one float4 per particle, a
// 6-color saturated palette cycled by particle index) so the same scene checks that one color per
// primitive reaches EVERY render path -- pick the raster ones with PT_PATH=impostor or
// PT_PATH=mesh. Without it the whole cloud keeps default_color.
#include <mimir/mimir.hpp>

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp (PT_LIGHT)
#include <vector>

using namespace mimir;

int main(int argc, char** argv)
{
    ViewerOptions opts;
    opts.render_mode      = RenderMode::Headless;
    // PT_LIGHT selects the shading path over the identical scene: path-traced (default) or the
    // Sphere3D raster impostor, so a per-primitive color source can be compared across both.
    const char* rp = getenv("PT_PATH");
    if (rp == nullptr) { rp = getenv("PT_LIGHT"); } // pre-rename spelling, still accepted
    const bool impostor = (rp != nullptr
        && (strcmp(rp, "impostor") == 0 || strcmp(rp, "phong") == 0));
    const bool mesh = (rp != nullptr
        && (strcmp(rp, "mesh") == 0 || strcmp(rp, "phong-mesh") == 0));
    opts.render_path      = impostor ? RenderPath::Impostor
                          : mesh ? RenderPath::Mesh
                          : RenderPath::PathTraced;
    opts.window.size      = { 512, 512 };
    opts.background_color  = { 0.10f, 0.10f, 0.13f, 1.f }; // dark grey: exercises env fill light
    opts.light_pos         = { -0.4082f, 0.4082f, 0.8165f }; // match benchmark world sun (from behind the camera)
    opts.pt_samples_per_pixel = 16; // more samples to expose GI/shadow noise cleanly
    opts.pt_subdivisions      = 2;  // exercise BLAS tessellation (320 tris)
    opts.pt_max_bounces       = (argc > 2) ? (unsigned)atoi(argv[2]) : 4; // GI depth (argv[2])
    opts.present.enable_fps_limit = false;
    // Optional harness knobs: PT_SPP overrides samples/pixel (lower = noisier, to expose the
    // denoiser); PT_DENOISE=1 turns on the à-trous denoiser.
    if (const char* s = getenv("PT_SPP"))     { opts.pt_samples_per_pixel = (unsigned)atoi(s); }
    if (const char* d = getenv("PT_DENOISE")) { opts.pt_denoise = (atoi(d) != 0); }
    if (const char* l = getenv("PT_LOD"))     { opts.pt_lod_cells = (unsigned)atoi(l); } // exercise LOD PT

    InstanceHandle engine = nullptr;
    createInstance(opts, &engine);

    // An NxNxN grid of particles in [-1,1]^3, driving the dynamic TLAS. N defaults to 5;
    // PT_GRID_N overrides it (e.g. 564 -> 179.4M particles, crossing the 4 GiB AABB-buffer
    // boundary at ~179M where BDA address truncation corrupted the record stream).
    int N = 5;
    if (const char* g = getenv("PT_GRID_N")) { N = atoi(g); }
    const unsigned int n = (unsigned int)N * N * N;
    float* d_pos = nullptr;
    AllocHandle pos_alloc{};
    allocLinear(engine, (void**)&d_pos, sizeof(float) * 3 * n, &pos_alloc);

    std::vector<float> host(3 * n);
    if (getenv("PT_RANDOM"))
    {
        // Uniform random positions in [-1,1]^3: consecutive INDICES land far apart in space,
        // like the kmodal benchmark. This is what makes BDA record-splicing corruption visible
        // as giant spheres (a spliced grid record mixes two nearly identical positions and
        // stays small; a spliced random record spans the domain).
        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t j = 0; j < host.size(); ++j)
        {
            s += 0x9E3779B97F4A7C15ull;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= z >> 31;
            host[j] = -1.f + 2.f * float(z >> 40) / float(1 << 24);
        }
    }
    else
    {
        unsigned int idx = 0;
        for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x)
        {
            auto coord = [N](int i){ return -1.f + 2.f * (float(i) + 0.5f) / float(N); };
            host[3 * idx + 0] = coord(x);
            host[3 * idx + 1] = coord(y);
            host[3 * idx + 2] = coord(z);
            idx++;
        }
    }
    cudaMemcpy(d_pos, host.data(), sizeof(float) * 3 * n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // PT_COLORS: one color per primitive, cycling a saturated 6-color palette by particle index.
    // Saturated primaries make the check trivial to score from the PPM -- each rendered sphere must
    // land in its own hue family, which a single default_color scene can never produce.
    const bool per_primitive_colors = getenv("PT_COLORS") != nullptr;
    float* d_col = nullptr;
    AllocHandle col_alloc{};
    if (per_primitive_colors)
    {
        allocLinear(engine, (void**)&d_col, sizeof(float) * 4 * n, &col_alloc);
        static const float palette[6][3] = {
            {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f},
            {1.f, 1.f, 0.f}, {1.f, 0.f, 1.f}, {0.f, 1.f, 1.f},
        };
        std::vector<float> hcol(4 * n);
        for (unsigned int i = 0; i < n; ++i)
        {
            const float* c = palette[i % 6];
            hcol[4 * i + 0] = c[0];
            hcol[4 * i + 1] = c[1];
            hcol[4 * i + 2] = c[2];
            hcol[4 * i + 3] = 1.f;
        }
        cudaMemcpy(d_col, hcol.data(), sizeof(float) * 4 * n, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

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
        // World radius of each sphere, scaled with the grid spacing so overridden PT_GRID_N stays
        // a resolvable cloud instead of a solid block (N = 5 keeps the historical 0.12). PT_SIZE_FRAC
        // overrides the fraction, e.g. 0.1 for a see-through cloud that exposes interior geometry.
        .default_size  = (getenv("PT_SIZE_FRAC") ? (float)atof(getenv("PT_SIZE_FRAC")) : 0.3f)
                       * (2.f / float(N)),
    };
    if (per_primitive_colors)
    {
        desc.attributes[AttributeType::Color] = AttributeDescription{
            .source = col_alloc,
            .size   = n,
            .format = FormatDescription::make<float4>(),
        };
    }
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
