// Remote rendering server (rr-server).
//
// Renders the k-modal 3D point cloud (the same Ornstein-Uhlenbeck gaussian-mixture simulation the
// particles-kmodal-3d benchmark uses) headless on the GPU, encodes it (H.264 via NVENC when
// available, else raw), and streams it to one connected rr-client at a time over TCP or QUIC,
// applying the camera/pause control the client sends back. Serves successive clients (reconnect).
//
// Shading and geometry are selectable independently, so the remote client sees the SAME impostor,
// phong-mesh, or path-traced view of the simulation that particles-kmodal-3d renders locally.
//
// Run from build/samples/remote-rendering/:
//   ./rr-server [port] [width] [height] [point_count] [h264] [transport] [token] [options]
//     h264:      pass 1 to H.264-encode (needs a build with ffmpeg; -DMIMIR_ENABLE_REMOTE=ON)
//     transport: "tcp" (default, works everywhere incl. ssh -L) or "quic" (-DMIMIR_ENABLE_QUIC=ON)
//     token:     optional shared secret the client must present (empty = accept any client)

#include <cstdio>  // snprintf (byte-limit formatting)
#include <cstdlib> // setenv (pin to a GPU via CUDA_VISIBLE_DEVICES)
#include <chrono>  // throttled [sim-sort] timing print
#include <string> // std::stoul
#include <vector>

#include <spdlog/spdlog.h>

#include "kmodal_sim.cuh"

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

// GPU capability reporting (core-count tables, RT-core estimate, NVENC/NVDEC probe) moved into the
// library: mimir::queryGpuCapabilities() + mimir::gpuBanner() (see mimir/mimir.hpp).

// Parse a color as "G" (grey level) or "R,G,B" in [0,1].
static float3 parseColor(const std::string& v)
{
    float r = 0.f, g = 0.f, b = 0.f;
    if (std::sscanf(v.c_str(), "%f,%f,%f", &r, &g, &b) == 3) { return { r, g, b }; }
    float grey = std::stof(v);
    return { grey, grey, grey };
}

// A render path used to be one conflated value; it is now the pair (Shading, Geometry). The old
// spellings stay accepted on --render-path / --light-model and set BOTH axes.
struct RenderPair { Shading shading; Geometry geometry; };
static RenderPair parseRenderPair(const std::string& v)
{
    if (v == "flat" || v == "none" || v == "point") return { Shading::Unlit, Geometry::Sprite };
    if (v == "impostor" || v == "phong")            return { Shading::Phong, Geometry::Impostor };
    if (v == "mesh" || v == "phong-mesh")           return { Shading::Phong, Geometry::Mesh };
    if (v == "path-traced" || v == "path-tracing")  return { Shading::PathTraced, Geometry::Impostor };
    fprintf(stderr, "Unknown render path '%s' (use none|phong|phong-mesh|path-tracing; prefer the "
        "separate --shading and --geometry flags)\n", v.c_str());
    exit(EXIT_FAILURE);
}

// --shading: how surfaces are lit, independent of their shape.
static Shading parseShading(const std::string& v)
{
    if (v == "unlit" || v == "none" || v == "flat") return Shading::Unlit;
    if (v == "phong")                               return Shading::Phong;
    if (v == "path-traced" || v == "path-tracing")  return Shading::PathTraced;
    fprintf(stderr, "Unknown shading '%s' (use unlit|phong|path-traced)\n", v.c_str());
    exit(EXIT_FAILURE);
}

// Lower-case geometry name for user-facing messages.
static const char* getGeometryName(Geometry g)
{
    return g == Geometry::Sprite ? "sprite" : (g == Geometry::Impostor ? "impostor" : "mesh");
}

// --geometry: what each marker is, independent of how it is lit.
static Geometry parseGeometry(const std::string& v)
{
    if (v == "sprite" || v == "point" || v == "flat") return Geometry::Sprite;
    if (v == "impostor" || v == "sphere")             return Geometry::Impostor;
    if (v == "mesh")                                  return Geometry::Mesh;
    fprintf(stderr, "Unknown geometry '%s' (use sprite|impostor|mesh)\n", v.c_str());
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
        "              No fixed maximum -- bounded by GPU memory. Positions cost 12 B/particle;\n"
        "              path-tracing WITHOUT --lod adds 24 B/particle for AABBs (36 B total);\n"
        "              --lod and the raster shadings (unlit/phong) stay ~12 B/particle.\n"
        "              An over-memory count is rejected up front (before any Vulkan allocation).\n"
        "              Rough per-card ceiling: ~7.5 B particles on a 96 GB GPU (none/phong/PT+LOD);\n"
        "              ~2.6 B for path-tracing without LOD; more on larger-VRAM cards.\n"
        "  h264        1 = H.264 encoding (needs -DMIMIR_ENABLE_REMOTE=ON), 0 = raw (default: 0)\n"
        "  transport   tcp (default, works with ssh -L) | quic (needs -DMIMIR_ENABLE_QUIC=ON)\n"
        "  token       Shared secret the client must send  (default: empty = accept anyone)\n"
        "\n"
        "Options (named, order-independent):\n"
        "  --shading S        How surfaces are lit: unlit | phong | path-traced   (default: phong)\n"
        "                     path-traced needs an RT GPU and traces analytic spheres, so\n"
        "                     --geometry does not apply to it.\n"
        "  --geometry G       What each marker is: sprite (flat 2D discs, unlit by nature) |\n"
        "                     impostor (ray-sphere impostors) | mesh (instanced icospheres,\n"
        "                     --subdiv)                                 (default: impostor)\n"
        "  --render-path P    Pre-split shorthand setting BOTH axes (alias: --light-model):\n"
        "                     none/point, phong, phong-mesh, path-tracing.\n"
        "  --size S           Marker size: pixels (sprite) or /100 world radius (lit) (default: 5)\n"
        "  --specular-power P Blinn-Phong specular exponent under --shading phong; higher = tighter,\n"
        "                     sharper highlight (default: 32)\n"
        "  --ambient A        Ambient light strength under --shading phong (0 = pure black shadows;\n"
        "                     default: 0.05)\n"
        "  --lod N            Level of detail: N^3 voxel grid, one representative per occupied\n"
        "                     cell, for any --shading (0 = per-particle, default). N is\n"
        "                     capped so the N^3 accumulator fits ~half of free device VRAM;\n"
        "                     larger N needs more memory. Trades detail for speed. Under lit\n"
        "                     modes, --size scales the representative sphere's cell-fill radius;\n"
        "                     under none, --size is unaffected (still the point's pixel size).\n"
        "                     The reduction (clear/scatter/emit) runs on custom native-CUDA kernels\n"
        "                     by default (single-digit ms at hundreds of millions of particles).\n"
        "                     Set MIMIR_LOD_NO_CUDA=1 to force the Vulkan-compute fallback instead\n"
        "                     (used automatically when CUDA/Vulkan interop is unavailable); it is\n"
        "                     slower under heavy atomic contention but scales past 2^32 particles,\n"
        "                     unlike a CUB-based reduction, which is limited to a 32-bit item count.\n"
        "  --lod-placement P  Where each cell's representative sits: cell (default) = the cell's\n"
        "                     geometric center; centroid = the mass centroid of the cell's particles.\n"
        "                     cell (default) drops 3 int64 atomics/particle in the reduction, so it is\n"
        "                     much faster at huge N (the reduction is atomic-bound there) and needs 8x\n"
        "                     less accumulator VRAM; centroid gives slightly finer positions at that\n"
        "                     atomic cost (~11x slower reduction at 6e9). No-op without --lod.\n"
        "  --lod-shape S      LOD representative shape: voxel (default) = solid grid-aligned cubes\n"
        "                     (forces cell-center placement, ignores --size for extent); sphere =\n"
        "                     round spheres honouring --lod-placement and --size. Lit models only;\n"
        "                     --shading unlit always draws flat points. No-op without --lod.\n"
        "  --sort-every N     Re-sort particles by Morton cell every N sim steps (needs --lod; 0 =\n"
        "                     off, default). Physically reordering particles gives the LOD centroid\n"
        "                     scatter's warp-aggregation real same-cell adjacency to collapse instead\n"
        "                     of colliding across unrelated warps (particles are NOT stored in cluster/\n"
        "                     spatial order otherwise -- each one picks its cluster independently at\n"
        "                     init). Sorting every step roughly pays back what it costs; every ~8 steps\n"
        "                     amortizes it while positions haven't drifted far enough to matter yet\n"
        "                     (measured: ~5-8x faster scatter, ~2.5-3x higher end-to-end fps under\n"
        "                     heavy clustering). No-op with --lod-placement cell (no atomics to help).\n"
        "                     Costs 32 B/particle extra VRAM (radix sort scratch). Env override:\n"
        "                     MIMIR_SIM_SORT_EVERY.\n"
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
        "                       <P>-<YYYYMMDD>-rr-server-n<count>-lod<N>-<light>-c<client>-s<server>-<gpu>.csv\n"
        "                     (e.g. run1-...-rr-server-n6G-lod256-pt-...; columns time_s,frame,fps,\n"
        "                     steps_s,kbps,encode_ms), pairing with the client's file. Pair with the\n"
        "                     client's --benchmark scripted camera.\n"
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
        "  --pause-at N       Freeze the simulation when it reaches step N and hold (default: 0 =\n"
        "                     disabled). Unlike --max-steps (which exits), the server keeps serving\n"
        "                     the frozen frame so a client can connect and capture. Use to grab\n"
        "                     identical screenshots of the same step across LOD configs.\n"
        "  --fly              First-person camera instead of the default trackball: the client\n"
        "                     looks with mouse-drag and flies with WASD (forward follows the gaze),\n"
        "                     good for touring inside a large scene. Default (no --fly) is the\n"
        "                     trackball (drag = orbit the scene, right-drag = zoom). The client\n"
        "                     adapts automatically (told via the stream handshake).\n"
        "  --dev N            GPU device id to run on (multi-GPU hosts)         (default: 0)\n"
        "                     Pins the process to that GPU via CUDA_VISIBLE_DEVICES before any\n"
        "                     CUDA/Vulkan init, so the render, encode and interop all land there.\n"
        "  --subdiv N         Icosphere tessellation 0=20 1=80 2=320 tris     (default: 1)\n"
        "                     --geometry mesh only -- path tracing uses analytic procedural AABB\n"
        "                     spheres, not tessellated meshes, so --subdiv has no effect there.\n"
        "  Path-tracing only (--shading path-traced):\n"
        "  --spp N            Samples per pixel per frame (antialiasing)      (default: 1)\n"
        "  --bounces N        Max path depth                                  (default: 4)\n"
        "  --bvh-rebuild-interval N  Full BLAS rebuild every N dirty frames, cheap in-place\n"
        "                     refits in between (1 = disable refit, rebuild every frame)\n"
        "                     (default: 8). Larger N trades traversal quality for speed as the\n"
        "                     scene deforms between rebuilds. Env override: MIMIR_PT_REBUILD_INTERVAL.\n"
        "  --denoise          Denoise each frame before display/encode; also makes the\n"
        "                     stream H.264-friendly at low bitrates (temporally stable)\n"
        "\n"
        "Examples:\n"
        "  # Minimal -- raw, TCP, phong impostors:\n"
        "  %s\n"
        "\n"
        "  # H.264, 1920x1080, 200k points, phong meshes:\n"
        "  %s 9000 1920 1080 200000 1 --geometry mesh\n"
        "\n"
        "  # Path-traced over QUIC:\n"
        "  %s 9000 1920 1080 50000 1 quic --shading path-traced --spp 2\n"
        "\n"
        "  # Paired benchmark for a research run: server and client each write a CSV that share\n"
        "  # the SAME <prefix> and line up for the identical 60 s scripted camera. On this host:\n"
        "  %s 9000 1280 720 100000000 1 tcp --benchmark run1\n"
        "  # then on the client:  rr-client 127.0.0.1 9000 \"\" tcp --benchmark run1\n"
        "  # -> run1-<date>-rr-server-...csv (here) + run1-<date>-rr-client-...csv (client);\n"
        "  #    plot both with research/scripts/plot_benchmark.py\n"
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
        prog, prog, prog, prog, prog, prog);
}

int main(int argc, char *argv[])
{
    // Defaults.
    unsigned short port      = 9000;
    int width                = 1280;
    int height               = 720;
    uint64_t point_count     = 100000;
    bool use_h264            = false;
    remote::TransportKind transport = remote::TransportKind::Tcp;
    std::string token        = "";
    Shading  shading         = Shading::Phong;
    Geometry geometry        = Geometry::Impostor;
    float size_px            = 5.f;
    float spec_power         = 32.f;  // --specular-power: Blinn-Phong exponent (phong/phong-mesh)
    float ambient_str        = 0.05f; // --ambient: ambient light strength (impostor/mesh)
    float3 pcolor            = { 0.82f, 0.82f, 0.88f };
    float3 background        = { 0.1f, 0.1f, 0.12f };
    PointsParams pts{};
    pts.count = point_count;
    unsigned int pt_spp     = 1;
    unsigned int pt_bounces = 4;
    unsigned int pt_subdiv  = 1;
    bool subdiv_set         = false;
    unsigned int pt_rebuild_interval = 8;
    bool pt_denoise         = false;
    unsigned int lod_cells   = 0;
    bool lod_centroid       = false;  // LOD placement: cell-center (default) vs centroid (opt-in)
    bool lod_voxel_shape    = true;   // --lod-shape voxel|sphere: LOD as cubes (default) vs spheres
    bool size_set           = false;  // whether --size was explicitly passed (for the voxel-mode ignore note)
    int sort_every_cli      = 0;      // periodic Morton spatial sort cadence, 0 = off (--sort-every)
    bool fly                = false;
    int bitrate_kbps        = 8000;
    int fps_cap             = 0;
    int steps_per_frame     = 0;
    int cuda_dev            = 0;
    size_t max_steps        = 0;
    size_t pause_at         = 0;
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
            if      (a == "--shading")   shading  = parseShading(v);
            else if (a == "--geometry")  geometry = parseGeometry(v);
            // Pre-split spellings: one value that sets both axes at once.
            else if (a == "--render-path" || a == "--light-model")
            { auto p = parseRenderPair(v); shading = p.shading; geometry = p.geometry; }
            else if (a == "--size")      { size_px = std::stof(v); size_set = true; }
            else if (a == "--specular-power") spec_power = std::stof(v);
            else if (a == "--ambient")        ambient_str = std::stof(v);
            else if (a == "--pcolor")      pcolor = parseColor(v);
            else if (a == "--background")  background = parseColor(v);
            else if (a == "--seed")        pts.seed = (uint32_t)std::stoul(v);
            else if (a == "--k")           pts.k = (unsigned int)std::stoul(v);
            else if (a == "--epsilon")     pts.epsilon = std::stof(v);
            else if (a == "--spp")         pt_spp = (unsigned int)std::stoul(v);
            else if (a == "--bounces")     pt_bounces = (unsigned int)std::stoul(v);
            else if (a == "--lod")         lod_cells = (unsigned int)std::stoul(v);
            else if (a == "--lod-placement") lod_centroid = (v == "centroid"); // default cell-center; centroid is opt-in
            else if (a == "--lod-shape") {
                if      (v == "voxel")  lod_voxel_shape = true;
                else if (v == "sphere") lod_voxel_shape = false;
                else { fprintf(stderr, "Unknown --lod-shape '%s' (use voxel|sphere)\n", v.c_str()); return EXIT_FAILURE; }
            }
            else if (a == "--sort-every")  sort_every_cli = std::stoi(v);
            else if (a == "--subdiv")    { pt_subdiv = (unsigned int)std::stoul(v); subdiv_set = true; }
            else if (a == "--bvh-rebuild-interval") pt_rebuild_interval = (unsigned int)std::stoul(v);
            else if (a == "--bitrate")     bitrate_kbps = std::stoi(v);
            else if (a == "--benchmark")   bench_csv = v;
            else if (a == "--fps")         fps_cap = std::stoi(v);
            else if (a == "--steps-per-frame") steps_per_frame = std::stoi(v);
            else if (a == "--dev")         cuda_dev = std::stoi(v);
            else if (a == "--max-steps")   max_steps = (size_t)std::stoull(v);
            else if (a == "--pause-at")    pause_at = (size_t)std::stoull(v);
            else { fprintf(stderr, "Unknown option %s\n\n", a.c_str()); usage(argv[0]); return EXIT_FAILURE; }
        }
        else { posv.push_back(a); }
    }
    if (posv.size() >= 1) port        = (unsigned short)std::stoi(posv[0]);
    if (posv.size() >= 3) { width = std::stoi(posv[1]); height = std::stoi(posv[2]); }
    if (posv.size() >= 4) {
        unsigned long long pc = std::stoull(posv[3]);
        if (pc == 0ull) {
            fprintf(stderr, "rr-server: point_count must be >= 1\n");
            return EXIT_FAILURE;
        }
        point_count = (uint64_t)pc;   // no upper cap: the memory pre-flight below bounds it
    }
    if (posv.size() >= 5) use_h264    = std::stoi(posv[4]) != 0;
    if (posv.size() >= 6) transport   = (posv[5] == "quic")?
        remote::TransportKind::Quic : remote::TransportKind::Tcp;
    if (posv.size() >= 7) token       = posv[6];
    pts.count = point_count;

    // Mesh spheres default to a smoother tessellation than path tracing's default; --subdiv still
    // overrides (matches particles-kmodal-3d/benchmark_mimir).
    if (geometry == Geometry::Mesh && !subdiv_set) { pt_subdiv = 2; }

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
    // GPU banner via the library (core tables, RT-core estimate, NVENC/NVDEC probe). Query device 0
    // (the single visible device after CUDA_VISIBLE_DEVICES pinned --dev N); print it as cuda_dev.
    auto caps = mimir::queryGpuCapabilities(0);
    if (!caps.name.empty()) {
        printf("rr-server: using GPU %s\n", mimir::gpuBanner(cuda_dev, caps).c_str());
    } else {
        printf("rr-server: using GPU device %d\n", cuda_dev);
    }

    // Periodic Morton spatial sort (--sort-every N, env override MIMIR_SIM_SORT_EVERY): parsed here so
    // its VRAM is in the pre-flight. The onesweep radix uses 32 B/particle (keys 8 + vals 8 ping-pong +
    // pos 12 + id 4 gather shadow) plus a flat decoupled-look-back status array (numTiles*256*8 B); see
    // kmodal_sim.cu SortScratch. Re-sorting gives the LOD centroid scatter's warp-aggregation real
    // same-cell adjacency to collapse (see lod_reduce.cu) -- sorting every step pays back roughly what
    // it costs, but every ~8 steps amortizes the sort while positions haven't drifted far enough to
    // matter yet (measured: ~5-8x faster scatter, ~2.5-3x higher end-to-end fps under heavy clustering).
    const char* sort_env = std::getenv("MIMIR_SIM_SORT_EVERY");
    int sort_every = sort_env ? std::atoi(sort_env) : sort_every_cli;
    const bool sort_on = (sort_every > 0 && lod_cells > 0);
    const unsigned long long sort_status_bytes = sort_on
        ? ((point_count + 2047ull) / 2048ull) * 256ull * 8ull + 16ull * 1024ull : 0ull; // status + gHist/gOff

    // Memory pre-flight: reject a count that will not fit the GPU memory free right now, BEFORE Vulkan
    // OOMs. Per-particle device allocations: mimir's interop (positions, always; + per-particle AABBs
    // under path-tracing WITHOUT LOD) plus the kmodal sim's per-particle cluster id (4 B, always --
    // see kmodal_sim.cu createClusters); + 32 B/particle for the spatial-sort radix when enabled.
    // The N^3 LOD accumulator and render targets are checked below.
    const bool pt_no_lod = (shading == Shading::PathTraced) && (lod_cells == 0);
    unsigned long long bytes_per_particle =
        mimir::interopBytesPerParticle(shading, lod_cells > 0) + 4ull  // +4: kmodal cluster id
        + (sort_on ? 32ull : 0ull);                                        // +32: radix sort scratch
    // Native (no-LOD) path tracing also builds a BVH acceleration structure -- BLAS storage plus build
    // scratch -- that the interop/position buffers counted above do NOT include, and it is large. A
    // measured 100M native-PT scene (single BLAS chunk) built 10.55 GB of BVH (3.35 storage + 7.20
    // scratch) => ~104 B/particle, and its total post-setup VRAM was 14.2 GB, matching the (40+104)
    // B/particle used here. Fold this in so an over-large native-PT scene is rejected up front with a
    // clear message instead of aborting mid-build on a Vulkan OOM. The estimate deliberately does NOT
    // amortise the shared build scratch across chunks (scratch is sized for the largest single chunk,
    // so per-particle BVH cost is actually a bit lower once N exceeds one chunk ~5.4e8) -- erring high
    // is the safe direction for a memory guard. The exact, driver-queried BVH size is logged at setup
    // (raytracing.cpp createDynamicBlasChunks). LOD path tracing rasterises the reduced cells into a
    // tiny AS, so this only applies to the no-LOD case.
    const unsigned long long as_bytes_per_particle = pt_no_lod ? 104ull : 0ull;  // BVH (BLAS+scratch) est.
    bytes_per_particle += as_bytes_per_particle;
    auto budget = mimir::memoryBudget(point_count, bytes_per_particle, 0);
    const size_t vram_free0 = budget.free_bytes; // reused by the LOD-accumulator check below
    {
        const unsigned long long need = (unsigned long long)point_count * bytes_per_particle + sort_status_bytes;
        if (need > (unsigned long long)vram_free0) {
            fprintf(stderr, "rr-server: %llu particles need %.1f GB (%s, %llu B/particle%s) but only %.1f GB "
                    "is free on the GPU right now -- max feasible here is ~%llu particles\n",
                    (unsigned long long)point_count, (double)need/1e9,
                    pt_no_lod ? "positions+ids+AABBs+BVH est." : "positions+ids", bytes_per_particle,
                    sort_on ? "+sort" : "", (double)vram_free0/1e9,
                    (unsigned long long)((vram_free0 > sort_status_bytes ? vram_free0 - sort_status_bytes : 0) / bytes_per_particle));
            return EXIT_FAILURE;
        }
        printf("rr-server: memory pre-flight OK -- %llu particles need %.1f GB of %.1f GB free "
               "(%llu B/particle%s); this GPU fits ~%llu particles in %s mode\n",
               (unsigned long long)point_count, (double)need/1e9, (double)vram_free0/1e9,
               bytes_per_particle, sort_on ? "+sort shadow" : "", (unsigned long long)budget.max_particles,
               pt_no_lod ? "path-tracing (no LOD)" : "raster/LOD");
    }

    // Accumulator is N^3 * bytes_per_cell. Reject an --lod whose accumulator would not fit the device
    // memory ACTUALLY free at this moment (vram_free0, queried live above via cudaMemGetInfo -- no
    // fixed budget fraction). Centroid placement keeps a per-cell count (u32) + a 3*u64 position sum
    // (~32 B/cell); cell-center keeps only the count (4 B/cell), so it fits a much larger N.
    const unsigned long long bytes_per_cell = lod_centroid ? 32ull : 4ull;
    const unsigned long long max_cells = (unsigned long long)vram_free0 / bytes_per_cell;
    // Largest N whose accumulator fits the currently-free VRAM, bounded by the uint32 cell-index limit.
    // The LOD shaders (pathtrace_lod_scatter.slang, pathtrace_lod_emit.slang) compute the linear cell
    // index and total cell count in 32-bit uint (total = gridN^3, lin = cx + gridN*(cy + gridN*cz)),
    // only safe while N^3 < 2^32 -- so this 1625 clamp is a CORRECTNESS bound (1625^3 < 2^32 <= 1626^3),
    // not a memory heuristic, and never allows an N that silently overflows the shader's occupancy math.
    unsigned int max_n = 0;
    while (max_n < 1625u &&
           (unsigned long long)(max_n + 1) * (max_n + 1) * (max_n + 1) <= max_cells) { ++max_n; }
    if (lod_cells > max_n) {
        fprintf(stderr, "rr-server: --lod %u needs %.1f GB for its N^3 accumulator but only %.1f GB is "
                        "free on the GPU right now; max feasible N is %u\n", lod_cells,
                        ((double)lod_cells*lod_cells*lod_cells*bytes_per_cell)/1e9,
                        (double)vram_free0/1e9, max_n);
        return EXIT_FAILURE;
    }
    if (lod_cells > 0 && (unsigned long long)lod_cells*lod_cells*lod_cells > (unsigned long long)point_count/8ull) {
        fprintf(stderr, "rr-server: note: --lod %u gives little benefit here; occupied cells approach "
                        "the particle count (%llu)\n", lod_cells, (unsigned long long)point_count);
    }
    // Raster light models (none/phong/phong-mesh) draw the WHOLE particle cloud as vertices/instances
    // every frame -- there is no BVH, no culling, nothing analogous to path-tracing's per-pixel cost.
    // Without --lod, a single vkCmdDraw at huge N can run long enough that the driver's own hang
    // detection kills it (observed: 1e9 points, phong, no --lod -> Xid 109 CTX SWITCH TIMEOUT ->
    // VK_ERROR_DEVICE_LOST, seconds after the client connects). path-tracing does not need this
    // warning: RT-core BVH traversal only touches primitives actual camera rays hit, so its cost
    // scales with screen pixels, not point_count. The threshold below is a heuristic headroom margin
    // under the observed failure, not a hard GPU limit -- it depends on GPU, resolution and topology.
    constexpr unsigned long long kRasterNoLodWarnThreshold = 200'000'000ull;
    if (shading != Shading::PathTraced && lod_cells == 0
        && point_count > kRasterNoLodWarnThreshold)
    {
        const char* path_name = getGeometryName(geometry);
        fprintf(stderr, "rr-server: warning: %llu particles with --geometry %s and no --lod draws "
                        "the entire cloud as unculled vertices every frame; this can run long enough "
                        "for the GPU driver to kill it (VK_ERROR_DEVICE_LOST). Add --lod N (e.g. 128) "
                        "to reduce to one representative per occupied cell, or use --shading "
                        "path-traced instead (its cost scales with screen pixels, not particle count).\n",
                        (unsigned long long)point_count, path_name);
    }

    ViewerOptions options;
    options.window.title      = "Mimir - remote kmodal-3d";
    options.render_mode       = RenderMode::Headless;
    options.window.size       = { width, height };
    options.shading           = shading;
    options.geometry          = geometry;
    // Same sun/material setup as particles-kmodal-3d/benchmark_mimir so the remote stream is lit
    // identically to the local benchmark: unit-length world-space direction TO the light, and a
    // 0.75-grey light color that matches the datoviz baseline's effective diffuse.
    options.light_pos         = { -0.4082f, 0.4082f, 0.8165f };
    options.light_color       = { 0.75f, 0.75f, 0.75f };
    options.specular_power    = spec_power;
    options.ambient_strength  = ambient_str;
    options.background_color   = { background.x, background.y, background.z, 1.f };
    options.pt_samples_per_pixel = pt_spp;
    options.pt_max_bounces       = pt_bounces;
    options.mesh_subdivisions    = pt_subdiv;
    options.pt_rebuild_interval  = pt_rebuild_interval;
    options.pt_denoise           = pt_denoise;
    options.pt_lod_cells         = lod_cells;
    options.lod_centroid         = lod_centroid;
    options.lod_voxel            = lod_voxel_shape;
    // Cell-center forcing for voxel LOD now happens in the engine (prepare), so every caller is
    // consistent; here just note that --size is ignored for the cube extent when voxels are active.
    if (lod_voxel_shape && lod_cells > 0 && shading != Shading::Unlit && size_set)
    {
        fprintf(stdout, "rr-server: LOD voxels ignore --size; cubes always fill the cell\n");
    }
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
    float size = (geometry == Geometry::Sprite) ? size_px : size_px / 100.f;

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
    // Path tracing packs particles as AABB sphere primitives, split across ceil(N/maxPrimitiveCount)
    // BLASes (one TLAS instance each), all read/written by buffer-device-address. So neither
    // maxStorageBufferRange, maxInstanceCount, nor the per-BLAS maxPrimitiveCount caps the particle
    // count -- only how many 24 B AABBs fit in one buffer (maxBufferSize/24) and, in practice first,
    // VRAM (~135 B/particle: positions 12 + AABBs 24 + BVH+scratch ~100).
    unsigned long long pt_cap = lim.max_buffer_size ? lim.max_buffer_size / 24ull : ~0ull;
    // Format a byte limit as GB, or "n/a" (unreported) / "unlimited" (driver sentinel >= 2^56, e.g.
    // UINT64_MAX for maxMemoryAllocationSize) so the line stays readable.
    auto gbstr = [](uint64_t bytes) {
        char b[24];
        if (bytes == 0)                snprintf(b, sizeof(b), "n/a");
        else if (bytes >= (1ull << 56)) snprintf(b, sizeof(b), "unlimited");
        else snprintf(b, sizeof(b), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
        return std::string(b);
    };
    // Practical ceiling: ~135 B/particle (positions 12 + AABBs 24 + BVH+scratch ~100) against total
    // VRAM. This, not the buffer cap, is what runs out first -- report both.
    const unsigned long long vram_particles = vram_total1 / 135ull;
    printf("rr-server: RT limits -- max-primitives/BLAS %llu (chunked), max-buffer %s => path-tracing "
           "buffer cap ~%.0f M particles; VRAM budget ~%llu M particles (~135 B each)\n",
           static_cast<unsigned long long>(lim.max_primitive_count),
           gbstr(lim.max_buffer_size).c_str(),
           pt_cap / 1e6,
           vram_particles / 1000000ull);

    const bool quic = (transport == remote::TransportKind::Quic);
    const char *shading_name = shading == Shading::Unlit ? "unlit"
                             : shading == Shading::Phong ? "phong" : "path-traced";
    printf("rr-server: %s port %u (%dx%d, %llu points, %s, %s %s). Connect with rr-client.\n",
        quic ? "UDP/QUIC" : "TCP", port, width, height, (unsigned long long)point_count,
        use_h264 ? "H.264" : "raw", shading_name, getGeometryName(geometry));

    // Experimental: periodically re-sort particles by Morton cell key so the render-side LOD reduction
    // gets spatial locality (warp-agg centroid scatter collapses ~3.3x faster). Parsed + budgeted above
    // (MIMIR_SIM_SORT_EVERY / sort_on); allocate the scratch and start sorted here. See kmodal_sim.cu.
    SortScratch sort_scratch{};
    if (sort_on)
    {
        sort_scratch = createSortScratch(point_count, lod_cells);
        printf("rr-server: sim spatial sort ON -- every %d steps, sortN=%u (Morton), scratch %.2f GB\n",
               sort_every, lod_cells, (double)sortScratchBytes(sort_scratch) / 1e9);
        launchSpatialSort(d_pos, (unsigned int*)clusters.ids, point_count, sort_scratch); // start sorted
        checkCuda(cudaDeviceSynchronize());
    }
    else if (sort_every > 0)
    {
        printf("rr-server: --sort-every ignored (needs --lod)\n");
        sort_every = 0;
    }
    uint64_t sim_step = 0;
    cudaEvent_t se0{}, se1{}; cudaEventCreate(&se0); cudaEventCreate(&se1);
    auto sort_log = std::chrono::steady_clock::now() - std::chrono::hours(1);

    // Blocks serving clients. The lambda advances the simulation each (non-paused) frame over
    // interop-mapped memory.
    serveRemote(instance, port, [&]{
        launchIntegrate3D(d_pos, point_count, clusters, rng);
        if (sort_every > 0 && (++sim_step % (uint64_t)sort_every == 0))
        {
            cudaEventRecord(se0);
            launchSpatialSort(d_pos, (unsigned int*)clusters.ids, point_count, sort_scratch);
            cudaEventRecord(se1); cudaEventSynchronize(se1);
            float ms = 0.f; cudaEventElapsedTime(&ms, se0, se1);
            // Surface this as its own "sort" pipeline stage in the server [stats] log and the client
            // HUD (see mimir::reportSortTimeNs) -- otherwise it silently inflates "compute" with no
            // visibility into why every sort_every-th step is so much more expensive than the rest.
            mimir::reportSortTimeNs(instance, static_cast<uint64_t>(ms * 1.0e6));
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - sort_log).count() >= 1000)
            {
                printf("[sim-sort] step %llu | spatial sort %.2f ms (sortN=%u, every %d)\n",
                       (unsigned long long)sim_step, ms, lod_cells, sort_every);
                sort_log = now;
            }
        }
        checkCuda(cudaDeviceSynchronize());
    }, max_steps, use_h264, transport, token.c_str(), bitrate_kbps,
        bench_csv.empty() ? nullptr : bench_csv.c_str(), fps_cap, steps_per_frame, pause_at);

    if (sort_scratch.sortN) destroySortScratch(sort_scratch);
    destroyClusters(clusters);
    destroyRngStates(rng);
    destroyInstance(instance);
    // d_pos is an interop allocation owned by mimir; freed via destroyInstance.
    return EXIT_SUCCESS;
}
