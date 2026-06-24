// Remote rendering server (rr-server).
//
// Renders a live 3D point cloud headless (no window) on the GPU, encodes it (H.264 via NVENC when
// available, else raw), and streams it to one connected rr-client at a time over TCP or QUIC,
// applying the camera/pause control the client sends back. Serves successive clients (reconnect).
//
// Run from build/samples/:
//   ./rr-server [port] [width] [height] [point_count] [h264] [transport] [token]
//     h264:      pass 1 to H.264-encode (needs a build with ffmpeg; -DMIMIR_ENABLE_REMOTE=ON)
//     transport: "tcp" (default, works everywhere incl. ssh -L) or "quic" (-DMIMIR_ENABLE_QUIC=ON)
//     token:     optional shared secret the client must present (empty = accept any client)

#include <curand_kernel.h>
#include <string> // std::stoul

#include <spdlog/spdlog.h>

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

__global__ void initRng(curandState *states, unsigned int count, unsigned int seed)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < count) { curand_init(seed, tidx, 0, &states[tidx]); }
}

__global__ void initPos(float *coords, size_t point_count, curandState *rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state = rng[tidx];
    for (auto i = tidx; i < point_count; i += stride)
    {
        points[i] = { curand_uniform(&state), curand_uniform(&state), curand_uniform(&state) };
    }
    rng[tidx] = state;
}

__global__ void integrate(float *coords, size_t point_count, curandState *rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state = rng[tidx];
    for (auto i = tidx; i < point_count; i += stride)
    {
        auto p = points[i];
        p.x += 0.05f * curand_normal(&state);
        p.y += 0.05f * curand_normal(&state);
        p.z += 0.05f * curand_normal(&state);
        p.x = fminf(fmaxf(p.x, 0.f), 1.f);
        p.y = fminf(fmaxf(p.y, 0.f), 1.f);
        p.z = fminf(fmaxf(p.z, 0.f), 1.f);
        points[i] = p;
    }
    rng[tidx] = state;
}

static void usage(const char *prog)
{
    printf(
        "Usage: %s [port] [width] [height] [point_count] [h264] [transport] [token]\n"
        "\n"
        "  port        TCP/UDP port to listen on          (default: 9000)\n"
        "  width       Render width in pixels             (default: 1280)\n"
        "  height      Render height in pixels            (default: 720)\n"
        "  point_count Number of 3-D points to simulate  (default: 10000)\n"
        "  h264        1 = H.264 encoding (needs -DMIMIR_ENABLE_REMOTE=ON), 0 = raw (default: 0)\n"
        "  transport   tcp (default, works with ssh -L) | quic (needs -DMIMIR_ENABLE_QUIC=ON)\n"
        "  token       Shared secret the client must send (default: empty = accept anyone)\n"
        "\n"
        "All arguments are positional and optional; omitted trailing args use their defaults.\n"
        "\n"
        "Examples:\n"
        "  # Minimal — listen on 9000, 1280x720, 10000 points, raw, TCP:\n"
        "  %s\n"
        "\n"
        "  # H.264, 1920x1080, 50000 points, TCP (good default for LAN / ssh -L):\n"
        "  %s 9000 1920 1080 50000 1\n"
        "\n"
        "  # Same but over QUIC (direct UDP, not through an ssh tunnel):\n"
        "  %s 9000 1920 1080 50000 1 quic\n"
        "\n"
        "  # H.264, QUIC, with a shared secret so only authorised clients can connect:\n"
        "  %s 9000 1920 1080 50000 1 quic mysecret\n"
        "\n"
        "Run from the build directory (shaders must be next to the binary):\n"
        "  cd samples/remote-rendering/build && ./rr-server ...\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }

    unsigned short port      = 9000;
    int width                = 1280;
    int height               = 720;
    unsigned int point_count = 10000;
    bool use_h264            = false;
    remote::TransportKind transport = remote::TransportKind::Tcp;
    std::string token        = "";
    if (argc >= 2) port        = static_cast<unsigned short>(std::stoi(argv[1]));
    if (argc >= 4) { width = std::stoi(argv[2]); height = std::stoi(argv[3]); }
    if (argc >= 5) point_count = std::stoul(argv[4]);
    if (argc >= 6) use_h264    = std::stoi(argv[5]) != 0;
    if (argc >= 7) transport   = (std::string(argv[6]) == "quic")?
        remote::TransportKind::Quic : remote::TransportKind::Tcp;
    if (argc >= 8) token       = argv[7];

    checkCuda(cudaSetDevice(0));

    const unsigned int block_size = 256;
    const unsigned int grid_size  = (point_count + block_size - 1) / block_size;
    const unsigned int rng_count  = grid_size * block_size;

    ViewerOptions options;
    options.render_mode = RenderMode::Headless;
    options.window.size = { width, height };
    options.background_color = { 0.1f, 0.1f, 0.12f, 1.f };
    options.light_pos = { 0.f, 0.f, 10.f };
    InstanceHandle instance = nullptr;
    createInstance(options, &instance);
    // createInstance silences logging in release builds; re-enable info so the server reports
    // what the transport and encoder are doing (waiting/connected, codec, zero-copy, etc.).
    spdlog::set_level(spdlog::level::info);

    float *d_coords       = nullptr;
    curandState *d_states = nullptr;
    AllocHandle points;
    allocLinear(instance, (void**)&d_coords, sizeof(float3) * point_count, &points);
    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * rng_count));

    initRng<<<grid_size, block_size>>>(d_states, rng_count, 12345u);
    checkCuda(cudaDeviceSynchronize());
    initPos<<<grid_size, block_size>>>(d_coords, point_count, d_states);
    checkCuda(cudaDeviceSynchronize());

    ViewDescription desc;
    desc.layout = Layout::make(point_count);
    desc.domain = DomainType::Domain3D;
    desc.type   = ViewType::Markers;
    desc.attributes[AttributeType::Position] = {
        .source = points,
        .size   = point_count,
        .format = FormatDescription::make<float3>(),
    };
    desc.default_size = 0.01f;
    desc.linewidth = 0.f;
    ViewHandle view = nullptr;
    createView(instance, &desc, &view);
    setCameraPosition(instance, {-.5f, -.5f, -3.f});

    const bool quic = (transport == remote::TransportKind::Quic);
    printf("rr-server: %s port %u (%dx%d, %u points, %s). Connect with rr-client.\n",
        quic ? "UDP/QUIC" : "TCP", port, width, height, point_count, use_h264 ? "H.264" : "raw");

    // Blocks serving clients. The lambda advances the simulation each (non-paused) frame over
    // interop-mapped memory.
    serveRemote(instance, port, [&]{
        integrate<<<grid_size, block_size>>>(d_coords, point_count, d_states);
        checkCuda(cudaDeviceSynchronize());
    }, 0, use_h264, transport, token.c_str());

    destroyInstance(instance);
    checkCuda(cudaFree(d_states));
    checkCuda(cudaFree(d_coords));
    return EXIT_SUCCESS;
}
