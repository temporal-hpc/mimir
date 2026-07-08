// Remote rendering server (rr-server).
//
// Renders the k-modal 3D point cloud (the same Ornstein-Uhlenbeck gaussian-mixture simulation the
// particles-kmodal-3d benchmark uses) headless on the GPU, encodes it (H.264 via NVENC when
// available, else raw), and streams it to one connected rr-client at a time over TCP or QUIC,
// applying the camera/pause control the client sends back. Serves successive clients (reconnect).
//
// The render path is selectable with --light-model, so the remote client sees the SAME phong,
// phong-mesh, or path-traced view of the simulation that particles-kmodal-3d renders locally.
//
// Run from build/samples/remote-rendering/:
//   ./rr-server [port] [width] [height] [point_count] [h264] [transport] [token] [options]
//     h264:      pass 1 to H.264-encode (needs a build with ffmpeg; -DMIMIR_ENABLE_REMOTE=ON)
//     transport: "tcp" (default, works everywhere incl. ssh -L) or "quic" (-DMIMIR_ENABLE_QUIC=ON)
//     token:     optional shared secret the client must present (empty = accept any client)

#include <string> // std::stoul
#include <vector>

#include <spdlog/spdlog.h>

#include "kmodal_sim.cuh"

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

// Parse a color as "G" (grey level) or "R,G,B" in [0,1].
static float3 parseColor(const std::string& v)
{
    float r = 0.f, g = 0.f, b = 0.f;
    if (std::sscanf(v.c_str(), "%f,%f,%f", &r, &g, &b) == 3) { return { r, g, b }; }
    float grey = std::stof(v);
    return { grey, grey, grey };
}

// Parse --light-model point|none|phong|phong-mesh|path-tracing into the instance-wide LightModel.
static LightModel parseLightModel(const std::string& v)
{
    if (v == "point")        return LightModel::None;
    if (v == "none")         return LightModel::None;
    if (v == "phong")        return LightModel::Phong;
    if (v == "phong-mesh")   return LightModel::PhongMesh;
    if (v == "path-tracing") return LightModel::PathTracing;
    fprintf(stderr, "Unknown --light-model '%s' (use point|none|phong|phong-mesh|path-tracing)\n", v.c_str());
    exit(EXIT_FAILURE);
}

static void usage(const char *prog)
{
    printf(
        "Usage: %s [port] [width] [height] [point_count] [h264] [transport] [token] [options]\n"
        "\n"
        "Positional (in order; omitted trailing args use their defaults):\n"
        "  port        TCP/UDP port to listen on           (default: 9000)\n"
        "  width       Render width in pixels              (default: 1280)\n"
        "  height      Render height in pixels             (default: 720)\n"
        "  point_count Number of 3-D points to simulate    (default: 100000)\n"
        "  h264        1 = H.264 encoding (needs -DMIMIR_ENABLE_REMOTE=ON), 0 = raw (default: 0)\n"
        "  transport   tcp (default, works with ssh -L) | quic (needs -DMIMIR_ENABLE_QUIC=ON)\n"
        "  token       Shared secret the client must send  (default: empty = accept anyone)\n"
        "\n"
        "Options (named, order-independent):\n"
        "  --light-model M    point/none = unlit discs, phong = lit sphere impostors,\n"
        "                     phong-mesh = lit instanced icosphere meshes (--subdiv),\n"
        "                     path-tracing = Vulkan RT              (default: phong)\n"
        "  --size S           Marker size: pixels (none) or /100 world radius (lit) (default: 5)\n"
        "  --pcolor C         Particle color: grey 'G' or 'R,G,B' in [0,1]    (default: light grey)\n"
        "  --background C     Window/sky color: grey 'G' or 'R,G,B' in [0,1]  (default: 0.1,0.1,0.12)\n"
        "  --seed N           RNG seed for positions/walk                     (default: 12345)\n"
        "  --k N              Gaussian modes (clusters) at init               (default: 8)\n"
        "  --epsilon E        Per-axis stddev of each mode                    (default: 0.05)\n"
        "  --bitrate N        H.264 target bitrate in kbps (h264 = 1 only)    (default: 8000)\n"
        "                     Path tracing without --denoise is temporally noisy and needs\n"
        "                     much more (e.g. 40000+) or interiors smear/ghost under motion.\n"
        "  --benchmark F      Write per-second server telemetry to CSV file F\n"
        "                     (time_s,frame,fps,kbps,encode_ms). Pair with the client's\n"
        "                     --benchmark scripted camera to replicate runs across servers.\n"
        "  --fps N            Cap the streamed session at N fps and honor --bitrate at that\n"
        "                     cadence (default: 0 = uncapped; the session runs at the natural\n"
        "                     render+encode+send rate, paced only by the link and client, and\n"
        "                     the wire rate scales with the achieved fps).\n"
        "  Path-tracing only (--light-model path-tracing):\n"
        "  --spp N            Samples per pixel per frame (antialiasing)      (default: 1)\n"
        "  --bounces N        Max path depth                                  (default: 4)\n"
        "  --subdiv N         Icosphere tessellation 0=20 1=80 2=320 tris     (default: 1)\n"
        "  --denoise          Denoise each frame before display/encode; also makes the\n"
        "                     stream H.264-friendly at low bitrates (temporally stable)\n"
        "\n"
        "Examples:\n"
        "  # Minimal -- raw, TCP, phong:\n"
        "  %s\n"
        "\n"
        "  # H.264, 1920x1080, 200k points, phong-mesh:\n"
        "  %s 9000 1920 1080 200000 1 --light-model phong-mesh\n"
        "\n"
        "  # Path-traced over QUIC:\n"
        "  %s 9000 1920 1080 50000 1 quic --light-model path-tracing --spp 2\n"
        "\n"
        "Run from the build directory (shaders must be next to the binary):\n"
        "  cd samples/remote-rendering/build && ./rr-server ...\n",
        prog, prog, prog, prog);
}

int main(int argc, char *argv[])
{
    // Defaults.
    unsigned short port      = 9000;
    int width                = 1280;
    int height               = 720;
    unsigned int point_count = 100000;
    bool use_h264            = false;
    remote::TransportKind transport = remote::TransportKind::Tcp;
    std::string token        = "";
    LightModel light_model   = LightModel::Phong;
    float size_px            = 5.f;
    float3 pcolor            = { 0.82f, 0.82f, 0.88f };
    float3 background        = { 0.1f, 0.1f, 0.12f };
    PointsParams pts{};
    pts.count = point_count;
    unsigned int pt_spp     = 1;
    unsigned int pt_bounces = 4;
    unsigned int pt_subdiv  = 1;
    bool subdiv_set         = false;
    bool pt_denoise         = false;
    int bitrate_kbps        = 8000;
    int fps_cap             = 0;
    std::string bench_csv   = "";

    // Split argv into positional (port width height ...) and named (--opt value) tokens. The
    // seven historical positional args stay compatible with the earlier rr-server CLI/README.
    std::vector<std::string> posv;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return EXIT_SUCCESS; }
        if (a == "--denoise") { pt_denoise = true; continue; } // flag, takes no value
        if (a.rfind("--", 0) == 0)
        {
            if (i + 1 >= argc)
            { fprintf(stderr, "Missing value for %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
            std::string v = argv[++i];
            if      (a == "--light-model") light_model = parseLightModel(v);
            else if (a == "--size")        size_px = std::stof(v);
            else if (a == "--pcolor")      pcolor = parseColor(v);
            else if (a == "--background")  background = parseColor(v);
            else if (a == "--seed")        pts.seed = (uint32_t)std::stoul(v);
            else if (a == "--k")           pts.k = (unsigned int)std::stoul(v);
            else if (a == "--epsilon")     pts.epsilon = std::stof(v);
            else if (a == "--spp")         pt_spp = (unsigned int)std::stoul(v);
            else if (a == "--bounces")     pt_bounces = (unsigned int)std::stoul(v);
            else if (a == "--subdiv")    { pt_subdiv = (unsigned int)std::stoul(v); subdiv_set = true; }
            else if (a == "--bitrate")     bitrate_kbps = std::stoi(v);
            else if (a == "--benchmark")   bench_csv = v;
            else if (a == "--fps")         fps_cap = std::stoi(v);
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { posv.push_back(a); }
    }
    if (posv.size() >= 1) port        = (unsigned short)std::stoi(posv[0]);
    if (posv.size() >= 3) { width = std::stoi(posv[1]); height = std::stoi(posv[2]); }
    if (posv.size() >= 4) point_count = (unsigned int)std::stoul(posv[3]);
    if (posv.size() >= 5) use_h264    = std::stoi(posv[4]) != 0;
    if (posv.size() >= 6) transport   = (posv[5] == "quic")?
        remote::TransportKind::Quic : remote::TransportKind::Tcp;
    if (posv.size() >= 7) token       = posv[6];
    pts.count = point_count;

    // Mesh spheres default to a smoother tessellation than path tracing's default; --subdiv still
    // overrides (matches particles-kmodal-3d/benchmark_mimir).
    if (light_model == LightModel::PhongMesh && !subdiv_set) { pt_subdiv = 2; }

    checkCuda(cudaSetDevice(0));

    ViewerOptions options;
    options.window.title      = "Mimir - remote kmodal-3d";
    options.render_mode       = RenderMode::Headless;
    options.window.size       = { width, height };
    options.light_model       = light_model;
    // Same sun/material setup as particles-kmodal-3d/benchmark_mimir so the remote stream is lit
    // identically to the local benchmark: unit-length world-space direction TO the light, and a
    // 0.75-grey light color that matches the datoviz baseline's effective diffuse.
    options.light_pos         = { -0.4082f, 0.4082f, 0.8165f };
    options.light_color       = { 0.75f, 0.75f, 0.75f };
    options.background_color   = { background.x, background.y, background.z, 1.f };
    options.pt_samples_per_pixel = pt_spp;
    options.pt_max_bounces       = pt_bounces;
    options.pt_subdivisions      = pt_subdiv;
    options.pt_denoise           = pt_denoise;
    // Match datoviz/particles-kmodal-3d framing of the [-1,1]^3 domain (45 deg vertical FOV).
    options.camera_fov        = 45.f;

    InstanceHandle instance = nullptr;
    createInstance(options, &instance);
    // createInstance silences logging in release builds; re-enable info so the server reports
    // what the transport and encoder are doing (waiting/connected, codec, zero-copy, etc.).
    spdlog::set_level(spdlog::level::info);

    // Interop position buffer (CUDA writes, Vulkan reads the same memory each frame).
    float *d_pos = nullptr;
    AllocHandle pos_alloc{};
    allocLinear(instance, (void**)&d_pos, sizeof(float3) * point_count, &pos_alloc);

    // K-modal simulation state (shared with particles-kmodal-3d).
    auto rng      = createRngStates(pts.seed);
    auto clusters = createClusters(pts);
    launchInitPositions(d_pos, pts, clusters, rng);

    // In None mode markers are pixel-sized point sprites; lit modes draw world-space spheres whose
    // default_size is a world-unit radius (/100), same convention as benchmark_mimir.
    float size = (light_model == LightModel::None) ? size_px : size_px / 100.f;

    ViewDescription desc{
        .type       = ViewType::Markers,
        .options    = MarkerOptions::defaults(),
        .domain     = DomainType::Domain3D,
        .attributes = {
            { AttributeType::Position, AttributeDescription{
                .source = pos_alloc,
                .size   = point_count,
                .format = FormatDescription::make<float3>(),
            }}
        },
        .layout        = Layout::make(point_count),
        .default_color = { pcolor.x, pcolor.y, pcolor.z, 1.f },
        .default_size  = size,
        .linewidth     = 0.f,
        .scale         = { 1.f, 1.f, 1.f },
    };
    ViewHandle view = nullptr;
    createView(instance, &desc, &view);

    // Points live in [-1,1]^3; the orbit view uses position as a scene translation, so z=-4 pushes
    // the scene back 4 units = effective eye at (0,0,+4) looking at the origin (datoviz home view).
    setCameraPosition(instance, { 0.f, 0.f, -4.f });

    const bool quic = (transport == remote::TransportKind::Quic);
    const char *lm =
        light_model == LightModel::None       ? "none" :
        light_model == LightModel::Phong      ? "phong" :
        light_model == LightModel::PhongMesh  ? "phong-mesh" : "path-tracing";
    printf("rr-server: %s port %u (%dx%d, %u points, %s, %s). Connect with rr-client.\n",
        quic ? "UDP/QUIC" : "TCP", port, width, height, point_count, use_h264 ? "H.264" : "raw", lm);

    // Blocks serving clients. The lambda advances the simulation each (non-paused) frame over
    // interop-mapped memory.
    serveRemote(instance, port, [&]{
        launchIntegrate3D(d_pos, point_count, clusters, rng);
        checkCuda(cudaDeviceSynchronize());
    }, 0, use_h264, transport, token.c_str(), bitrate_kbps,
        bench_csv.empty() ? nullptr : bench_csv.c_str(), fps_cap);

    destroyClusters(clusters);
    destroyRngStates(rng);
    destroyInstance(instance);
    // d_pos is an interop allocation owned by mimir; freed via destroyInstance.
    return EXIT_SUCCESS;
}
