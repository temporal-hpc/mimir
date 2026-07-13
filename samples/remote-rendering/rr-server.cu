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

#include <cstdio>  // snprintf (NVML PCI bus id)
#include <cstdlib> // setenv (pin to a GPU via CUDA_VISIBLE_DEVICES)
#include <cstring> // std::strstr (GPU-name matching)
#include <string> // std::stoul
#include <vector>

#include <nvml.h>  // NVENC/NVDEC presence for the startup GPU banner
#include <spdlog/spdlog.h>

#include "kmodal_sim.cuh"

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

// Shader cores per SM by compute capability (NVIDIA's well-known table, from CUDA samples).
static int coresPerSM(int major, int minor)
{
    switch ((major << 4) | minor)
    {
        case 0x30: case 0x32: case 0x35: case 0x37: return 192;            // Kepler
        case 0x50: case 0x52: case 0x53:            return 128;            // Maxwell
        case 0x60:                                  return 64;             // Pascal GP100
        case 0x61: case 0x62:                       return 128;            // Pascal
        case 0x70: case 0x72: case 0x75:            return 64;             // Volta / Turing
        case 0x80:                                  return 64;             // Ampere GA100 (A100)
        case 0x86: case 0x87: case 0x89:            return 128;            // Ampere / Ada
        case 0x90:                                  return 128;            // Hopper
        default:                                    return 128;            // newer archs (approx)
    }
}

// Tensor cores per SM: none before Volta, 8 on Volta/Turing, 4 on Ampere and later.
static int tensorPerSM(int major) { return major < 7 ? 0 : (major == 7 ? 8 : 4); }

// RT-core count is NOT queryable via CUDA (or CUDA-visible), so this is a best-effort estimate:
// NVIDIA has ~1 RT core per SM on RT-capable GPUs, and 0 on datacenter compute parts. Those are
// hard to tell apart by compute capability alone (a new datacenter arch can share a CC with a
// consumer one), so match the known compute families by name too.
static int rtCores(const cudaDeviceProp& p)
{
    static const char *const compute_only[] = {
        "V100", "A100", "A800", "H100", "H200", "H800", "GH200",
        "B100", "B200", "B300", "GB200", "GB300", // Blackwell datacenter: no RT cores
    };
    for (const char *s : compute_only) { if (std::strstr(p.name, s) != nullptr) { return 0; } }
    const bool cc_datacenter =
        (p.minor == 0 && (p.major == 7 || p.major == 8 || p.major == 9 || p.major == 10));
    const bool rt_capable = (p.major * 10 + p.minor) >= 75 && !cc_datacenter; // Turing+ w/ display
    return rt_capable ? p.multiProcessorCount : 0;
}

// Query NVENC/NVDEC presence via NVML. NVML ignores CUDA_VISIBLE_DEVICES and enumerates every
// physical GPU, so find the one whose PCI location matches our CUDA device (numeric compare, no
// bus-string formatting). Encoder-capacity / decoder-utilization return NOT_SUPPORTED on a GPU
// that lacks that engine -- e.g. the A100 (NVDEC yes, NVENC no).
static void queryVideoEngines(const cudaDeviceProp& p, bool& nvenc, bool& nvdec)
{
    nvenc = nvdec = false;
    if (nvmlInit_v2() != NVML_SUCCESS) { return; }
    unsigned int count = 0;
    if (nvmlDeviceGetCount_v2(&count) == NVML_SUCCESS)
    {
        for (unsigned int i = 0; i < count; ++i)
        {
            nvmlDevice_t dev{};
            nvmlPciInfo_t pci{};
            if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS) { continue; }
            if (nvmlDeviceGetPciInfo_v3(dev, &pci) != NVML_SUCCESS)      { continue; }
            if (static_cast<int>(pci.domain) != p.pciDomainID ||
                static_cast<int>(pci.bus)    != p.pciBusID    ||
                static_cast<int>(pci.device) != p.pciDeviceID) { continue; }
            unsigned int cap = 0;
            nvenc = nvmlDeviceGetEncoderCapacity(dev, NVML_ENCODER_QUERY_H264, &cap) == NVML_SUCCESS;
            unsigned int util = 0, period = 0;
            nvdec = nvmlDeviceGetDecoderUtilization(dev, &util, &period) == NVML_SUCCESS;
            break;
        }
    }
    nvmlShutdown();
}

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
        "  --benchmark P      Write per-second server telemetry to an auto-named CSV. P is a\n"
        "                     path+prefix; the full name is assembled when a client connects as\n"
        "                       <P>-<YYYYMMDD>-rr-server-c<client>-s<server>-<gpu>.csv\n"
        "                     (columns time_s,frame,fps,steps_s,kbps,encode_ms), pairing with the\n"
        "                     client's file. Pair with the client's --benchmark scripted camera.\n"
        "  --fps N            Cap the streamed FRAME rate at N fps and honor --bitrate at that\n"
        "                     cadence (default: 0 = uncapped). The simulation runs on its own\n"
        "                     thread, so this caps pixels-on-the-wire only and never slows the\n"
        "                     sim's steps/s. Uncapped over TCP trades latency for throughput:\n"
        "                     frames flood reliable buffers faster than the link drains, so they\n"
        "                     queue (bufferbloat) — set --fps (e.g. 30/60) over an SSH tunnel.\n"
        "  --steps-per-frame N How the simulation couples to frame production      (default: 0)\n"
        "                     0 = decoupled: the sim runs on its own thread at full speed and\n"
        "                         each frame samples the latest state — a viewer never slows the\n"
        "                         run (monitoring). --fps caps pixels-on-the-wire only.\n"
        "                     N>=1 = lockstep: advance exactly N sim steps, then render one frame,\n"
        "                         sequentially (tear-free, deterministic — good for recording or\n"
        "                         reproducing). N=1 is the classic 1-step-per-frame mode. Here\n"
        "                         --fps (and a slow client) paces the SIM too, by design.\n"
        "  --max-steps N      Stop the simulation after N steps (default: 0 = run endlessly).\n"
        "                     Distinct from point_count: this is how many steps to run, not how\n"
        "                     many points. The remaining progress shows in the client's HUD\n"
        "                     (step x of y); with 0 the HUD reads 'iteration x of unlimited'.\n"
        "  --fly              First-person camera instead of the default trackball: the client\n"
        "                     looks with mouse-drag and flies with WASD (forward follows the gaze),\n"
        "                     good for touring inside a large scene. Default (no --fly) is the\n"
        "                     trackball (drag = orbit the scene, right-drag = zoom). The client\n"
        "                     adapts automatically (told via the stream handshake).\n"
        "  --dev N            GPU device id to run on (multi-GPU hosts)         (default: 0)\n"
        "                     Pins the process to that GPU via CUDA_VISIBLE_DEVICES before any\n"
        "                     CUDA/Vulkan init, so the render, encode and interop all land there.\n"
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
        "Serving to a remote client over SSH (e.g. this server in a Slurm + Pyxis job):\n"
        "  The server binds all interfaces (0.0.0.0) and enroot shares the host network, so it is\n"
        "  reachable at <compute-node-name>:<port> with no container port mapping. SSH forwards TCP\n"
        "  only, so run with transport 'tcp' and have the client tunnel in -- QUIC is UDP and will\n"
        "  NOT traverse an ssh -L tunnel. Find the node with 'squeue -u $USER' (NODELIST column).\n"
        "\n"
        "  # In the job (this server):\n"
        "  %s 9000 1280 720 100000 1 tcp\n"
        "  # On the client laptop: forward local 9000 -> this node via the login node, then run\n"
        "  # rr-client against the tunnel:\n"
        "  ssh -N -L 9000:<compute-node-name>:<port> <user>@<supercomputer-url>\n"
        "  rr-client 127.0.0.1 9000 \"\" tcp\n"
        "\n"
        "  Concrete example (node gpu042, cluster hpc.example.edu, port 9000):\n"
        "    ssh -N -L 9000:gpu042:9000 alice@hpc.example.edu\n"
        "\n"
        "Run from the build directory (shaders must be next to the binary):\n"
        "  cd samples/remote-rendering/build && ./rr-server ...\n",
        prog, prog, prog, prog, prog);
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
    bool fly                = false;
    int bitrate_kbps        = 8000;
    int fps_cap             = 0;
    int steps_per_frame     = 0;
    int cuda_dev            = 0;
    size_t max_steps        = 0;
    std::string bench_csv   = "";

    // Split argv into positional (port width height ...) and named (--opt value) tokens. The
    // seven historical positional args stay compatible with the earlier rr-server CLI/README.
    std::vector<std::string> posv;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return EXIT_SUCCESS; }
        if (a == "--denoise") { pt_denoise = true; continue; } // flag, takes no value
        if (a == "--fly")     { fly = true; continue; }        // first-person camera (flag)
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
            else if (a == "--steps-per-frame") steps_per_frame = std::stoi(v);
            else if (a == "--dev")         cuda_dev = std::stoi(v);
            else if (a == "--max-steps")   max_steps = (size_t)std::stoull(v);
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

    // Pin the process to the requested GPU before the first CUDA/Vulkan call: with only that
    // device visible, the engine's UUID-matched interop selection lands on it (and it becomes
    // CUDA device 0 within this process). Must precede any CUDA runtime call to take effect.
    if (cuda_dev < 0) { fprintf(stderr, "rr-server: --dev must be >= 0\n"); return EXIT_FAILURE; }
    setenv("CUDA_VISIBLE_DEVICES", std::to_string(cuda_dev).c_str(), /*overwrite=*/1);
    int visible_devs = 0;
    if (cudaGetDeviceCount(&visible_devs) != cudaSuccess || visible_devs < 1)
    {
        fprintf(stderr, "rr-server: GPU device %d is not available\n", cuda_dev);
        return EXIT_FAILURE;
    }
    checkCuda(cudaSetDevice(0)); // device 0 of the (now single-device) visible set == --dev N
    cudaDeviceProp gpu_prop{};
    if (cudaGetDeviceProperties(&gpu_prop, 0) == cudaSuccess)
    {
        bool nvenc = false, nvdec = false;
        queryVideoEngines(gpu_prop, nvenc, nvdec);
        printf("rr-server: using GPU device %d (%s) | %.0f GB | %d SMs | %d CUDA cores | "
               "%d tensor cores | %d RT cores | NVENC %s | NVDEC %s\n",
            cuda_dev, gpu_prop.name,
            static_cast<double>(gpu_prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
            gpu_prop.multiProcessorCount,
            gpu_prop.multiProcessorCount * coresPerSM(gpu_prop.major, gpu_prop.minor),
            gpu_prop.multiProcessorCount * tensorPerSM(gpu_prop.major),
            rtCores(gpu_prop), nvenc ? "yes" : "no", nvdec ? "yes" : "no");
    }
    else
    {
        printf("rr-server: using GPU device %d\n", cuda_dev);
    }
    // Baseline free VRAM (context already up) so we can report mimir's footprint after setup.
    size_t vram_free0 = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free0, &vram_total);

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
    // --fly: run the first-person camera. serveRemote seeds the fly pose and interprets the
    // client's mouse-look/WASD; the client is told via the Hello flags so it adapts its input.
    if (fly) { options.camera_control = CameraControl::Fly; }

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

    // VRAM footprint after setup: the drop in free memory since baseline captures everything mimir
    // put on the device -- the CUDA particle buffer plus the Vulkan render targets, geometry/BVH and
    // interop buffers (these share the same VRAM, so cudaMemGetInfo sees them too).
    size_t vram_free1 = 0, vram_total1 = 0;
    cudaMemGetInfo(&vram_free1, &vram_total1);
    const double toMB = 1.0 / (1024.0 * 1024.0);
    const double used_mb = (vram_free0 > vram_free1) ? (vram_free0 - vram_free1) * toMB : 0.0;
    const double part_mb = static_cast<double>(sizeof(float3)) * point_count * toMB;
    const double vk_gb = static_cast<double>(deviceLocalMemory(instance)) / (1024.0 * 1024.0 * 1024.0);
    printf("rr-server: Vulkan-visible VRAM %.0f GB | mimir uses %.0f MB (particles %.0f MB, "
           "render+geometry+interop %.0f MB)\n",
           vk_gb, used_mb, part_mb, used_mb > part_mb ? used_mb - part_mb : 0.0);

    // Per-buffer size limits. These, NOT free VRAM, are what fail vkCreateBuffer for the path-tracing
    // instance buffer (64 B/particle, needs both STORAGE and a device address) once it grows past a
    // couple GiB -- hence the OUT_OF_DEVICE_MEMORY at ~2^25 particles on a card with 260+ GB free. The
    // instance buffer is bound by the smaller of max-storage-range and max-buffer-size, so that ratio
    // over 64 B is the path-tracing particle ceiling on this GPU.
    const mimir::DeviceBufferLimits lim = mimir::deviceBufferLimits(instance);
    // Path tracing packs particles as AABB sphere primitives in one BLAS (one TLAS instance), so the
    // ceiling is the SMALLEST of: the BLAS primitive limit (maxPrimitiveCount, ~2^29 on NVIDIA); how
    // many 12 B positions fit in a storage-buffer binding (maxStorageBufferRange/12 -- the AABB writer
    // reads positions as a descriptor); and how many 24 B AABBs fit in one buffer (maxBufferSize/24;
    // the AABB buffer is buffer-device-address, so not storage-range limited). maxInstanceCount no
    // longer applies. Whichever bites first, VRAM may cap lower still.
    unsigned long long pt_cap = lim.max_primitive_count ? lim.max_primitive_count : ~0ull;
    if (lim.max_storage_buffer_range && lim.max_storage_buffer_range / 12ull < pt_cap)
    {
        pt_cap = lim.max_storage_buffer_range / 12ull;
    }
    if (lim.max_buffer_size && lim.max_buffer_size / 24ull < pt_cap)
    {
        pt_cap = lim.max_buffer_size / 24ull;
    }
    // Format a byte limit as GB, or "n/a" (unreported) / "unlimited" (driver sentinel >= 2^56, e.g.
    // UINT64_MAX for maxMemoryAllocationSize) so the line stays readable.
    auto gbstr = [](uint64_t bytes) {
        char b[24];
        if (bytes == 0)                snprintf(b, sizeof(b), "n/a");
        else if (bytes >= (1ull << 56)) snprintf(b, sizeof(b), "unlimited");
        else snprintf(b, sizeof(b), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
        return std::string(b);
    };
    printf("rr-server: RT limits -- max-primitives %llu, storage-range %s, max-buffer %s, max-alloc %s "
           "=> path-tracing caps at %llu particles (~%.0f M)\n",
           static_cast<unsigned long long>(lim.max_primitive_count),
           gbstr(lim.max_storage_buffer_range).c_str(), gbstr(lim.max_buffer_size).c_str(),
           gbstr(lim.max_memory_allocation_size).c_str(),
           pt_cap, pt_cap / 1e6);

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
    }, max_steps, use_h264, transport, token.c_str(), bitrate_kbps,
        bench_csv.empty() ? nullptr : bench_csv.c_str(), fps_cap, steps_per_frame);

    destroyClusters(clusters);
    destroyRngStates(rng);
    destroyInstance(instance);
    // d_pos is an interop allocation owned by mimir; freed via destroyInstance.
    return EXIT_SUCCESS;
}
