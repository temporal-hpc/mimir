// Headless Phase-1 verification for LightModel::PathTracing: renders the static icosphere
// grid to an offscreen frame and writes it to a PPM so the result can be inspected without a
// window. Not part of the benchmark; a throwaway acceptance check.
#include <mimir/mimir.hpp>

#include <cstdio>

using namespace mimir;

int main(int argc, char** argv)
{
    ViewerOptions opts;
    opts.render_mode      = RenderMode::Headless;
    opts.light_model      = LightModel::PathTracing;
    opts.window.size      = { 512, 512 };
    opts.background_color  = { 0.f, 0.f, 0.f, 1.f }; // black so any traced pixels stand out
    opts.present.enable_fps_limit = false;

    InstanceHandle engine = nullptr;
    createInstance(opts, &engine);

    // A minimal interop Markers view so the engine's uniform-buffer/view machinery has a view
    // to size against (the RT path traces the static grid regardless of these positions; the
    // buffer is never written). Mirrors the benchmark's view setup.
    constexpr unsigned int n = 64;
    float* d_pos = nullptr;
    AllocHandle pos_alloc{};
    allocLinear(engine, (void**)&d_pos, sizeof(float) * 3 * n, &pos_alloc);

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
        .default_color = { 1.f, 1.f, 1.f, 1.f },
        .default_size  = 0.02f,
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
