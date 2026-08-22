// Remote rendering client (rr-client): a thin native viewer for rr-server.
//
// Connects to rr-server, authenticates, receives the video stream (Hello + framed messages:
// frames or Stats telemetry), decodes H.264 (software decoder — a standard bitstream, so no GPU
// is needed on the client) or accepts raw, and displays it in a window while sending mouse/keyboard
// interaction back. Transport is auto by default: it tries QUIC (UDP + TLS + congestion control)
// and falls back to TCP if QUIC doesn't come up (UDP blocked, ssh -L tunnel, ...).
//
// Window behaviour: the window is freely resizable and the frame is simply stretched to fill it
// (it may look blurrier when enlarged). The client never asks the server to change resolution —
// that keeps a remote viewer simple and the server's render cost fixed.
//
// A minimal HUD overlay in the top-left corner (toggle with H) shows where the simulation runs
// (user@host:port, transport, codec), the end-to-end latency, the stream fps, the sim progress
// (step x of y, or unlimited), the scene size + LOD mode, and a pipeline breakdown of where a
// frame's time goes: compute + sort + render (local GPU cost) vs. encode + net + decode (remote
// transfer cost); sort is the optional periodic spatial re-sort (--sort-every), shown only when
// active. Other keys: P pauses the simulation, Q/Esc/Ctrl+W quit.
//
// Depends only on the wire-protocol header + ffmpeg + GLFW/OpenGL (no mimir, CUDA, or Vulkan):
// it models a laptop-class thin client. QUIC (ngtcp2 + OpenSSL) is an optional compile-time
// add-on: when the build can't find it, the viewer is TCP-only (see MIMIR_RRC_HAVE_QUIC below).
//
// Run from build/samples/:
//   ./rr-client [host] [port] [token] [auto|quic|tcp] [frames]
//     token:  shared secret (must match the server; empty if the server has none)
//     frames: if > 0, run headless — receive N frames, save rr-client.ppm, and exit (for testing
//             without a display); otherwise open an interactive window.
//
// To decode in hardware on an NVIDIA client instead, swap avcodec_find_decoder(AV_CODEC_ID_H264)
// for avcodec_find_decoder_by_name("h264_cuvid") below.

#include <mimir/remote_protocol.hpp>
#include "rr_input.hpp" // shared GLFW -> ControlKind input capture
using namespace mimir::remote;

// QUIC transport is optional: the whole ngtcp2/OpenSSL stack compiles in only when the build
// found it (MimirRemoteClient.cmake defines MIMIR_RRC_HAVE_QUIC). Without it the viewer is
// TCP-only, which lets it build on distros lacking libngtcp2_crypto_ossl (e.g. Ubuntu <= 24.04).
#ifdef MIMIR_RRC_HAVE_QUIC
#include <ngtcp2/ngtcp2.h>
#include <ngtcp2/ngtcp2_crypto.h>
#include <ngtcp2/ngtcp2_crypto_ossl.h>
// ngtcp2_conn_get_expiry2 was added after 1.22.x; alias it to the original on older releases.
#ifndef HAVE_NGTCP2_CONN_GET_EXPIRY2
#  define ngtcp2_conn_get_expiry2 ngtcp2_conn_get_expiry
#endif

// ngtcp2 1.22.0 added the "2"-suffixed get_new_connection_id / get_path_challenge_data callbacks
// and the ngtcp2_stateless_reset_token struct (pre-1.22 forms take a raw uint8_t* token). Support
// both so the viewer builds on older distro ngtcp2 (e.g. Ubuntu 26.04 ships 1.16.0).
#if defined(NGTCP2_VERSION_NUM) && NGTCP2_VERSION_NUM >= 0x011600
#  define MIMIR_NGTCP2_CALLBACKS2 1
#else
#  define MIMIR_NGTCP2_CALLBACKS2 0
#endif

#include <openssl/ssl.h>
#include <openssl/rand.h>
#endif // MIMIR_RRC_HAVE_QUIC

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <GLFW/glfw3.h>
#include <GL/gl.h>

// stb_truetype rasterizes the HUD text with anti-aliasing from a real font (single-header,
// public domain). font8x8.h stays as the guaranteed fallback when no system font is found.
#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>
#include "font8x8.h" // embedded 8x8 bitmap font for the HUD overlay (no text-rendering deps)

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{

#ifdef MIMIR_RRC_HAVE_QUIC
const unsigned char ALPN[] = {5, 'm', 'i', 'm', 'i', 'r'};
#endif

constexpr uint64_t NS_PER_SEC = 1000000000ull; // == ngtcp2's NGTCP2_SECONDS, without the dependency

uint64_t now_ns()
{
    timespec tp{};
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return static_cast<uint64_t>(tp.tv_sec) * NS_PER_SEC + static_cast<uint64_t>(tp.tv_nsec);
}

// Client steady clock in milliseconds since startup, as the 32-bit stamp carried by ControlMsg
// and echoed back in frames (0 is reserved for "no stamp", hence the +1).
uint32_t now_ms()
{
    static const uint64_t t0 = now_ns();
    return static_cast<uint32_t>((now_ns() - t0) / 1000000ull) + 1;
}

// ---------------------------------------------------------------------------------------------
// Shared state between the session thread (transport + decode) and the main thread (window/UI).
// ---------------------------------------------------------------------------------------------
struct Shared
{
    std::mutex frame_mtx;
    std::vector<unsigned char> latest; // most recent decoded BGRA frame
    int w = 0, h = 0;
    uint64_t frame_seq = 0;            // bumped on each new frame
    bool have_geometry = false;        // true once the first Hello/frame sets w,h

    std::mutex ctrl_mtx;
    std::deque<ControlMsg> outgoing;   // control events from the UI, awaiting send

    std::atomic<bool> running{true};   // session alive
    std::atomic<bool> quit{false};     // shutdown requested (UI closed / headless done)
    std::atomic<uint64_t> frames{0};   // total decoded frames (headless wait condition)
    std::atomic<uint32_t> ctrl_sent{0};// interaction events sent since the last stats window
};
Shared g;

// ---------------------------------------------------------------------------------------------
// Terminal colors for the client's console logs, so info/success/error stand out at a glance. Gated
// on stdout being a TTY, so redirected/piped output stays plain (no escape-code garbage in files).
namespace ansi
{
    inline bool on()          { static const bool e = ::isatty(fileno(stdout)); return e; }
    inline const char* grn()  { return on() ? "\033[1;32m" : ""; } // success (connected, ready)
    inline const char* red()  { return on() ? "\033[1;31m" : ""; } // errors
    inline const char* cyn()  { return on() ? "\033[36m"   : ""; } // info / notices
    inline const char* dim()  { return on() ? "\033[2m"    : ""; } // per-second stats (low priority)
    inline const char* rst()  { return on() ? "\033[0m"    : ""; }
}

// HUD: a minimal always-on overlay (toggle with H) answering "what am I looking at?": which
// server runs the simulation (user@host:port, transport, codec), the end-to-end latency, and
// the simulation progress (step x of y, or unlimited). Written by the session thread as
// Hello/Stats/latency samples arrive; read by the window thread each frame.
// ---------------------------------------------------------------------------------------------
struct Hud
{
    std::mutex mtx;
    std::string server;             // "user@host:port  QUIC/H.264 1280x720" (set on connect)
    double latency_ms = -1.0;       // smoothed end-to-end latency (-1 = no sample yet)
    double fps = 0.0;               // server-reported stream fps
    uint64_t step = 0;              // simulation steps completed (server lifetime)
    uint64_t step_limit = 0;        // steps the server stops at (0 = unlimited)
    uint64_t particle_count = 0;    // particles the sim advances (0 = older server, hidden in HUD)
    uint64_t particles_per_sec = 0; // current sim throughput (particle_count * steps/s)
    uint32_t lod_cells = 0;         // LOD mode from the Hello: 0 = native, N = N^3 voxel grid
    uint32_t steps_per_frame = 0;   // Hello: 0 = decoupled, N>=1 = lockstep (N steps/frame on path)
    // Real-time pipeline stage times, so the HUD shows where a frame's wall-clock goes: local
    // GPU cost (compute + render) vs. the remote transfer cost (encode + network + decode).
    double compute_ms = 0.0;        // server: mean sim compute() time per step
    double sort_ms = 0.0;           // server: mean spatial-sort sub-stage time per step (part of compute_ms; --sort-every)
    double render_ms = 0.0;         // server: mean GPU render/trace time per frame
    double lod_ms = 0.0;            // server: mean LOD-reduction time per frame (part of render_ms)
    double denoise_ms = 0.0;        // server: mean denoiser time per frame (part of render_ms; PT+--denoise)
    double encode_ms = 0.0;         // server: mean encode (or readback) time per frame
    double decode_ms = 0.0;         // client: mean decode time per frame (measured here)
    std::string gpu;                // server GPU model (own HUD line, with live VRAM)
    uint32_t vram_used_mb = 0;      // server GPU memory used / total (MiB), for the GPU HUD line
    uint32_t vram_total_mb = 0;
    uint32_t power_w = 0;           // server board power draw / limit (W), on the GPU HUD line
    uint32_t power_limit_w = 0;
    std::string bench_path;         // --benchmark: the CSV path, shown under the BENCHMARK banner
    bool have_stats = false;        // first Stats message arrived
    bool benchmark = false;         // --benchmark active: HUD shows a "BENCHMARK MODE" banner
    std::atomic<bool> visible{true};
};
Hud g_hud;

// Dialed address, for the HUD when the server doesn't announce its hostname (older server).
std::string g_dial_host, g_dial_port;
// Transport label of the established session ("TCP"/"QUIC"), for HUD refreshes on re-Hello.
const char *g_transport = "?";
// Initial window size the viewer opens at (--window W H). 0 = match the stream resolution.
// This is purely local: the frame is still stretched to fill the window (server res unchanged).
int g_win_w = 0, g_win_h = 0;
// How long to wait for the server's first frame before giving up (--first-frame-timeout SECONDS).
// The first path-traced frame is a cold full BLAS build + trace with no refit/accumulation shortcut,
// so it can take a minute-plus at hundreds of millions of particles and scales with N: ~62 s was
// observed at 2^29, so 2^30 can exceed 4 min. Default generously; a dead connection still exits
// early via g.running regardless of this cap.
int g_first_frame_timeout_sec = 300;
// Set from the Hello handshake: true when the server runs the Fly camera, so the viewer drives
// it with mouse-look (CameraLook) + WASD (CameraMove) instead of the trackball orbit/zoom/pan.
std::atomic<bool> g_fly{false};

void hud_set_server(const Hello& hello, const char *transport)
{
    g_fly.store((hello.flags & HELLO_CAMERA_FLY) != 0, std::memory_order_relaxed);
    // The wire strings are NUL-padded but not guaranteed NUL-terminated at full length.
    char user[USER_MAX + 1] = {}; std::memcpy(user, hello.user, USER_MAX);
    char host[HOST_MAX + 1] = {}; std::memcpy(host, hello.host, HOST_MAX);
    char gpu[GPU_MAX + 1]   = {}; std::memcpy(gpu, hello.gpu, GPU_MAX);
    std::string where;
    if (user[0]) { where += user; where += '@'; }
    where += host[0] ? host : g_dial_host;
    where += ':'; where += g_dial_port;
    char line[256];
    snprintf(line, sizeof(line), "%s  %s/%s %ux%u", where.c_str(), transport,
        static_cast<Codec>(hello.codec) == Codec::H264 ? "H.264" : "raw",
        hello.width, hello.height);
    std::lock_guard<std::mutex> lock(g_hud.mtx);
    g_hud.server = line;
    g_hud.gpu = gpu;                   // shown on its own HUD line with live VRAM (see hud_lines)
    g_hud.lod_cells = hello.lod_cells; // 0 = native, N = N^3 grid (shown on the particle line)
    g_hud.steps_per_frame = hello.steps_per_frame; // 0 = decoupled, N = lockstep (compute on path)
}

// Human-scaled count (K/M/G) and per-second rate for the particle HUD line, mirroring the
// server's formatParticleCount / formatParticleRate so both ends read the same.
std::string hud_scale_count(uint64_t n)
{
    char b[32];
    const double d = static_cast<double>(n);
    if      (n >= 1000000000ull) { snprintf(b, sizeof(b), "%.2f G", d / 1e9); }
    else if (n >= 1000000ull)    { snprintf(b, sizeof(b), "%.1f M", d / 1e6); }
    else if (n >= 1000ull)       { snprintf(b, sizeof(b), "%.1f K", d / 1e3); }
    else                         { snprintf(b, sizeof(b), "%llu", static_cast<unsigned long long>(n)); }
    return b;
}
std::string hud_scale_rate(uint64_t per_sec)
{
    char b[32];
    const double d = static_cast<double>(per_sec);
    if      (per_sec >= 1000000000ull) { snprintf(b, sizeof(b), "%.2f Gpart/s", d / 1e9); }
    else if (per_sec >= 1000000ull)    { snprintf(b, sizeof(b), "%.1f Mpart/s", d / 1e6); }
    else                               { snprintf(b, sizeof(b), "%llu part/s", static_cast<unsigned long long>(per_sec)); }
    return b;
}

// Composes the HUD text from the latest session state (two short lines, plus a particle line).
std::vector<std::string> hud_lines()
{
    std::lock_guard<std::mutex> lock(g_hud.mtx);
    std::vector<std::string> lines;
    lines.push_back(g_hud.server.empty() ? "connecting..." : g_hud.server);
    // GPU line: model | live server VRAM (used/total) | live board power, pipe-separated to match the
    // pipeline line. Each metric is appended only when the server reported it.
    if (!g_hud.gpu.empty())
    {
        char g[220];
        int n = snprintf(g, sizeof(g), "%s", g_hud.gpu.c_str());
        if (g_hud.vram_total_mb > 0 && n > 0 && n < (int)sizeof(g))
            n += snprintf(g + n, sizeof(g) - n, " | %.1f/%.1f GB",
                g_hud.vram_used_mb / 1024.0, g_hud.vram_total_mb / 1024.0);
        if (g_hud.power_limit_w > 0 && n > 0 && n < (int)sizeof(g))
            n += snprintf(g + n, sizeof(g) - n, " | %u/%u W", g_hud.power_w, g_hud.power_limit_w);
        else if (g_hud.power_w > 0 && n > 0 && n < (int)sizeof(g))
            n += snprintf(g + n, sizeof(g) - n, " | %u W", g_hud.power_w);
        lines.push_back(g);
    }
    // Particle line: scene size + LOD mode + sim throughput. Placed ABOVE the latency line. Only when
    // the server reported it (0 = older server without the fields, or a viewer-only session).
    if (g_hud.have_stats && g_hud.particle_count > 0)
    {
        char lodbuf[16];
        if (g_hud.lod_cells == 0) { snprintf(lodbuf, sizeof(lodbuf), "native"); }
        else { snprintf(lodbuf, sizeof(lodbuf), "%u^3", g_hud.lod_cells); }
        char l3[160];
        snprintf(l3, sizeof(l3), "%s particles | LOD %s | %s",
            hud_scale_count(g_hud.particle_count).c_str(), lodbuf,
            hud_scale_rate(g_hud.particles_per_sec).c_str());
        lines.push_back(l3);
    }
    char l2[192];
    if (!g_hud.have_stats)
    {
        snprintf(l2, sizeof(l2), "latency  --  ms e2e | --  fps | step --");
    }
    else if (g_hud.step_limit > 0)
    {
        snprintf(l2, sizeof(l2), "latency %5.1f ms e2e | %4.1f fps | step %llu of %llu (%.1f%%)",
            g_hud.latency_ms < 0 ? 0.0 : g_hud.latency_ms, g_hud.fps,
            static_cast<unsigned long long>(g_hud.step),
            static_cast<unsigned long long>(g_hud.step_limit),
            100.0 * static_cast<double>(g_hud.step) / static_cast<double>(g_hud.step_limit));
    }
    else
    {
        snprintf(l2, sizeof(l2), "latency %5.1f ms e2e | %4.1f fps | step %llu of unlimited",
            g_hud.latency_ms < 0 ? 0.0 : g_hud.latency_ms, g_hud.fps,
            static_cast<unsigned long long>(g_hud.step));
    }
    lines.push_back(l2);
    // Pipeline line, in pipeline order: compute -> sort -> lod -> render -> denoise -> encode -> network
    // -> decode. render + denoise + encode + network + decode are the components of the end-to-end
    // latency on the line above (render/denoise = local GPU cost; encode + network + decode = remote
    // transfer cost). `network~` is the residual latency minus the measured stages (wire transit +
    // frame-boundary wait), floored at 0. render_ms from the server INCLUDES lod and denoise, so those
    // are broken out as their own stages and render is shown as the pure draw/trace cost (render_ms -
    // lod - denoise); the full render_ms stays on the latency path so network~ is unchanged. Likewise
    // compute_ms INCLUDES the optional sort sub-stage (it runs inside compute(), see
    // mimir::reportSortTimeNs), so sort is broken out the same way and compute is shown pure (compute_ms
    // - sort_ms); the full compute_ms is what's on the latency path. compute (sim ms/step): in DECOUPLED
    // mode it is OFF the latency path (nothing subtracted); in LOCKSTEP it is on-path, so steps_per_frame
    // * compute_ms is subtracted from network~. sort/lod/denoise appear only when nonzero.
    if (g_hud.have_stats)
    {
        const double lat = g_hud.latency_ms < 0 ? 0.0 : g_hud.latency_ms;
        const double compute_on_path = g_hud.compute_ms * static_cast<double>(g_hud.steps_per_frame);
        const double compute_pure = std::max(0.0, g_hud.compute_ms - g_hud.sort_ms);
        const double render_pure = std::max(0.0, g_hud.render_ms - g_hud.lod_ms - g_hud.denoise_ms);
        const double network_ms = std::max(0.0,
            lat - g_hud.render_ms - g_hud.encode_ms - g_hud.decode_ms - compute_on_path);
        // Fixed-width numeric fields (matches the latency line's %5.1f/%4.1f above): every value here
        // updates live, so an unpadded %.1f/%.2f changes the STRING LENGTH the moment a value crosses a
        // digit boundary (e.g. network~ 9.3 -> 10.3 ms), and since the HUD panel is sized to the longest
        // current line every frame (hud_rasterize[_ttf]'s max_w), that reflows the whole panel width --
        // felt as a flicker. Padding keeps every field's width constant so only the digits change.
        char sortseg[36] = "", lodseg[32] = "", dnseg[32] = "";
        if (g_hud.sort_ms > 0.005)    { snprintf(sortseg, sizeof(sortseg), "sort %5.1f ms | ", g_hud.sort_ms); }
        if (g_hud.lod_ms > 0.005)     { snprintf(lodseg, sizeof(lodseg), "lod %5.1f ms | ", g_hud.lod_ms); }
        if (g_hud.denoise_ms > 0.005) { snprintf(dnseg, sizeof(dnseg), "denoise %5.1f ms | ", g_hud.denoise_ms); }
        char l4[320];
        snprintf(l4, sizeof(l4),
            "compute %6.2f ms/step | %s%srender %5.1f ms | %sencode %5.1f ms | network~%5.1f ms | decode %5.1f ms",
            compute_pure, sortseg, lodseg, render_pure, dnseg, g_hud.encode_ms, network_ms, g_hud.decode_ms);
        lines.push_back(l4);
    }
    return lines;
}

// Benchmark banner + CSV destination, drawn separately in the bottom-left corner (see hud_draw).
std::vector<std::string> bench_lines()
{
    std::lock_guard<std::mutex> lock(g_hud.mtx);
    std::vector<std::string> lines;
    if (g_hud.benchmark)
    {
        lines.push_back("== BENCHMARK MODE ==");
        if (!g_hud.bench_path.empty()) { lines.push_back("results in " + g_hud.bench_path); }
    }
    return lines;
}

// ---------------------------------------------------------------------------------------------
// TrueType HUD font: loaded once from a system monospace face so the overlay is crisp and
// anti-aliased instead of the blocky 2x-magnified 8x8 bitmap. If no font opens, `ok` stays false
// and hud_draw() falls back to the embedded bitmap (see hud_rasterize).
// ---------------------------------------------------------------------------------------------
struct HudFont
{
    std::vector<unsigned char> data;      // font file bytes (must outlive `info`)
    stbtt_fontinfo info{};
    int asc = 0, desc = 0, gap = 0;       // vertical metrics in raw font units (desc negative)
    bool ok = false;
};
HudFont g_font;

// The glyph pixel height tracks the window so the HUD keeps the same relative size when resized.
// px = framebuffer_height / HUD_PX_DIVISOR, so a 720-px-tall window yields ~16 px (the size the
// overlay was originally tuned at), clamped to a legible range.
constexpr float HUD_PX_DIVISOR = 45.f;
constexpr float HUD_PX_MIN = 11.f;
constexpr float HUD_PX_MAX = 48.f;

float hud_px_for_height(int fb_h)
{
    const float px = static_cast<float>(fb_h) / HUD_PX_DIVISOR;
    return std::clamp(px, HUD_PX_MIN, HUD_PX_MAX);
}

// Tries a list of common monospace TTFs across distros; the first that opens and parses wins.
// Monospace keeps the changing latency/fps/step numbers from jittering the layout each frame.
void hud_font_init()
{
    static const char *const candidates[] = {
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/LiberationMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    };
    for (const char *path : candidates)
    {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { continue; }
        const std::streamsize n = f.tellg();
        if (n <= 0) { continue; }
        g_font.data.resize(static_cast<size_t>(n));
        f.seekg(0);
        if (!f.read(reinterpret_cast<char*>(g_font.data.data()), n)) { g_font.data.clear(); continue; }
        const int off = stbtt_GetFontOffsetForIndex(g_font.data.data(), 0);
        if (off < 0 || !stbtt_InitFont(&g_font.info, g_font.data.data(), off))
        {
            g_font.data.clear();
            continue;
        }
        stbtt_GetFontVMetrics(&g_font.info, &g_font.asc, &g_font.desc, &g_font.gap);
        g_font.ok = true;
        printf("rr-client: HUD font = %s\n", path);
        return;
    }
    printf("rr-client: HUD font = 8x8 bitmap (no system monospace TTF found)\n");
}

// Rasterizes the HUD lines with the loaded TrueType font into an RGBA image: white, anti-aliased
// glyphs composited over a translucent dark backdrop (same output contract as hud_rasterize).
void hud_rasterize_ttf(const std::vector<std::string>& lines, float px,
    std::vector<unsigned char>& rgba, int& w, int& h)
{
    // Everything scales off the requested pixel height so the HUD grows/shrinks with the window.
    const float scale = stbtt_ScaleForPixelHeight(&g_font.info, px);
    const int ascent = static_cast<int>(std::lround(g_font.asc * scale));
    const int line_h = static_cast<int>(std::lround((g_font.asc - g_font.desc + g_font.gap) * scale));
    const int PAD = std::max(2, static_cast<int>(std::lround(px * 0.4f)));

    // Width = widest line's summed advance (+ kerning); height = one line_h per line.
    int max_w = 0;
    for (const auto& l : lines)
    {
        int pen = 0;
        for (size_t i = 0; i < l.size(); ++i)
        {
            int adv = 0, lsb = 0;
            const int c = static_cast<unsigned char>(l[i]);
            stbtt_GetCodepointHMetrics(&g_font.info, c, &adv, &lsb);
            pen += static_cast<int>(std::lround(adv * scale));
            if (i + 1 < l.size())
            {
                const int nc = static_cast<unsigned char>(l[i + 1]);
                pen += static_cast<int>(std::lround(
                    stbtt_GetCodepointKernAdvance(&g_font.info, c, nc) * scale));
            }
        }
        max_w = std::max(max_w, pen);
    }
    w = max_w + 2 * PAD;
    h = static_cast<int>(lines.size()) * line_h + 2 * PAD;

    // Start from an opaque-enough dark backdrop; glyphs are composited on top (source-over) so the
    // text stays crisp when the whole image is later alpha-blended over the video frame.
    rgba.assign(static_cast<size_t>(w) * h * 4, 0);
    for (size_t i = 3; i < rgba.size(); i += 4) { rgba[i] = 170; } // backdrop alpha

    std::vector<unsigned char> cov; // scratch coverage bitmap, reused per glyph
    for (size_t li = 0; li < lines.size(); ++li)
    {
        const std::string& l = lines[li];
        const int baseline = PAD + static_cast<int>(li) * line_h + ascent;
        int pen = PAD;
        for (size_t ci = 0; ci < l.size(); ++ci)
        {
            const int c = static_cast<unsigned char>(l[ci]);
            int adv = 0, lsb = 0;
            stbtt_GetCodepointHMetrics(&g_font.info, c, &adv, &lsb);
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            stbtt_GetCodepointBitmapBox(&g_font.info, c, scale, scale,
                &x0, &y0, &x1, &y1);
            const int gw = x1 - x0, gh = y1 - y0;
            if (gw > 0 && gh > 0)
            {
                cov.assign(static_cast<size_t>(gw) * gh, 0);
                stbtt_MakeCodepointBitmap(&g_font.info, cov.data(), gw, gh, gw,
                    scale, scale, c);
                for (int gy = 0; gy < gh; ++gy)
                {
                    const int dy = baseline + y0 + gy;
                    if (dy < 0 || dy >= h) { continue; }
                    for (int gx = 0; gx < gw; ++gx)
                    {
                        const int dx = pen + x0 + gx;
                        if (dx < 0 || dx >= w) { continue; }
                        const float sa = cov[static_cast<size_t>(gy) * gw + gx] / 255.f;
                        if (sa <= 0.f) { continue; }
                        // White source over the existing (dark, translucent) pixel.
                        unsigned char *d = &rgba[(static_cast<size_t>(dy) * w + dx) * 4];
                        const float da = d[3] / 255.f;
                        const float oa = sa + da * (1.f - sa);
                        const unsigned char rgb = static_cast<unsigned char>(
                            std::lround(oa > 0.f ? (sa / oa) * 255.f : 0.f));
                        d[0] = d[1] = d[2] = rgb;
                        d[3] = static_cast<unsigned char>(std::lround(oa * 255.f));
                    }
                }
            }
            pen += static_cast<int>(std::lround(adv * scale));
            if (ci + 1 < l.size())
            {
                const int nc = static_cast<unsigned char>(l[ci + 1]);
                pen += static_cast<int>(std::lround(
                    stbtt_GetCodepointKernAdvance(&g_font.info, c, nc) * scale));
            }
        }
    }
}

// Rasterizes text lines with the embedded 8x8 font into an RGBA image: white glyphs on a
// translucent dark backdrop, one 10-px row per line plus a small margin all around.
void hud_rasterize(const std::vector<std::string>& lines,
    std::vector<unsigned char>& rgba, int& w, int& h)
{
    constexpr int GLYPH = 8, LINE_H = 10, PAD = 5;
    size_t cols = 0;
    for (const auto& l : lines) { cols = std::max(cols, l.size()); }
    w = static_cast<int>(cols) * GLYPH + 2 * PAD;
    h = static_cast<int>(lines.size()) * LINE_H + 2 * PAD;
    rgba.assign(static_cast<size_t>(w) * h * 4, 0);
    for (size_t i = 3; i < rgba.size(); i += 4) { rgba[i] = 170; } // backdrop alpha
    for (size_t li = 0; li < lines.size(); ++li)
    {
        const int y0 = PAD + static_cast<int>(li) * LINE_H + 1;
        for (size_t ci = 0; ci < lines[li].size(); ++ci)
        {
            unsigned char c = static_cast<unsigned char>(lines[li][ci]);
            if (c < 0x20 || c > 0x7E) { c = '?'; }
            const unsigned char *glyph = FONT8X8[c - 0x20];
            const int x0 = PAD + static_cast<int>(ci) * GLYPH;
            for (int y = 0; y < 8; ++y)
            {
                for (int x = 0; x < 8; ++x)
                {
                    if (glyph[y] & (1u << x)) // bit N = pixel at x=N
                    {
                        unsigned char *p = &rgba[(static_cast<size_t>(y0 + y) * w + x0 + x) * 4];
                        p[0] = p[1] = p[2] = p[3] = 255;
                    }
                }
            }
        }
    }
}

// Draws the HUD into the top-left corner of the window (over the video frame). fb_h is the current
// framebuffer height, so the text can scale with the window.
void hud_draw(int fb_h)
{
    static std::vector<unsigned char> px;
    int w = 0, h = 0;
    // TrueType glyphs are rasterized at device resolution (draw 1:1) and sized to the window; the
    // bitmap fallback is a fixed 8 px and needs the historic 2x magnification to stay readable.
    const float zoom = g_font.ok ? 1.f : 2.f;
    if (g_font.ok) { hud_rasterize_ttf(hud_lines(), hud_px_for_height(fb_h), px, w, h); }
    else           { hud_rasterize(hud_lines(), px, w, h); }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPixelZoom(zoom, -zoom); // negative y draws downward from the raster position
    glRasterPos2f(-1.f, 1.f);
    glBitmap(0, 0, 0.f, 0.f, 8.f, -8.f, nullptr); // nudge in from the top-left corner, in window pixels
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

    // Benchmark banner + CSV path in the BOTTOM-left corner (its own block, so it stays put regardless
    // of how many status lines the main HUD has). Anchor at (-1,-1), then nudge the raster position up
    // by the block's height (+ an 8 px margin) so the block's bottom sits ~8 px above the window bottom.
    const auto bl = bench_lines();
    if (!bl.empty())
    {
        static std::vector<unsigned char> bpx;
        int bw = 0, bh = 0;
        if (g_font.ok) { hud_rasterize_ttf(bl, hud_px_for_height(fb_h), bpx, bw, bh); }
        else           { hud_rasterize(bl, bpx, bw, bh); }
        glRasterPos2f(-1.f, -1.f);
        glBitmap(0, 0, 0.f, 0.f, 8.f, static_cast<float>(bh) * zoom + 8.f, nullptr);
        glDrawPixels(bw, bh, GL_RGBA, GL_UNSIGNED_BYTE, bpx.data());
    }
    glDisable(GL_BLEND);
}

// Optional CSV time-series log (one row per server Stats window), for offline plotting.
FILE *g_csv = nullptr;
// --benchmark path+prefix; the full CSV name is assembled once the Hello names the server + GPU.
std::string g_bench_prefix;

// Opens the benchmark CSV once, naming it from this run's identities so it pairs with the
// server's file:  <prefix>-<date>-rr-client-c<client>-s<server>-<gpu>.csv
void bench_csv_open(const Hello& hello)
{
    if (g_csv || g_bench_prefix.empty()) { return; }
    char client[256]{}; gethostname(client, sizeof(client) - 1);
    char server[HOST_MAX + 1] = {}; std::memcpy(server, hello.host, HOST_MAX);
    char gpu[GPU_MAX + 1] = {}; std::memcpy(gpu, hello.gpu, GPU_MAX);
    const std::string path = benchmarkCsvPath(g_bench_prefix, "client", client,
        server[0] ? server : g_dial_host, gpu,
        hello.particle_count, hello.lod_cells, hello.render_path);
    g_csv = fopen(path.c_str(), "w");
    if (!g_csv) { fprintf(stderr, "cannot open csv log '%s'\n", path.c_str()); return; }
    fprintf(g_csv, "time_s,fps,kbps,server_ms,server_ms_std,compute_ms,render_ms,decode_ms,decode_ms_std,"
        "lat_mean_ms,lat_std_ms,lat_p50_ms,lat_p95_ms,lat_max_ms,lost,ctrl_events,phase\n");
    fflush(g_csv);
    { std::lock_guard<std::mutex> lock(g_hud.mtx); g_hud.bench_path = path; } // show it under the HUD banner
    printf("%srr-client: benchmark CSV -> %s%s\n", ansi::cyn(), path.c_str(), ansi::rst());
}

// Current interaction phase, logged as a CSV column so plots can shade idle vs. moving spans.
// The --benchmark script sets its phase names ("idle", "orbit", ...); outside benchmark mode it
// stays empty and the CSV derives "move"/"idle" from whether control events were sent.
std::atomic<const char*> g_phase{""};

void store_frame(const unsigned char *bgra, int w, int h)
{
    std::lock_guard<std::mutex> lock(g.frame_mtx);
    g.w = w; g.h = h;
    g.latest.assign(bgra, bgra + static_cast<size_t>(w) * h * 4);
    g.frame_seq++;
    g.have_geometry = true;
    g.frames++;
}

// Pushes a control event from the UI thread to be sent by the session thread. Every event is
// stamped with the client clock; the server echoes the newest stamp in the next frame, which is
// how end-to-end latency is measured. ControlKind::None events are pure latency heartbeats.
void ui_control(ControlKind kind, float a = 0.f, float b = 0.f)
{
    ControlMsg msg{};
    msg.kind = static_cast<uint8_t>(kind);
    msg.a = a; msg.b = b;
    msg.stamp_ms = now_ms();
    if (kind != ControlKind::None) { g.ctrl_sent++; }
    std::lock_guard<std::mutex> lock(g.ctrl_mtx);
    g.outgoing.push_back(msg);
}

// Queues a heartbeat at most every 50 ms so latency keeps being sampled without interaction.
void maybe_heartbeat(uint64_t& last_hb_ns)
{
    const uint64_t now = now_ns();
    if (now - last_hb_ns >= 50ull * 1000000ull)
    {
        ui_control(ControlKind::None);
        last_hb_ns = now;
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder: software H.264 (or raw passthrough) -> BGRA, written into the shared latest frame.
// ---------------------------------------------------------------------------------------------
struct Decoder
{
    const AVCodec *codec = nullptr;
    AVCodecContext *ctx = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    SwsContext *sws = nullptr;
    std::vector<unsigned char> tmp; // scratch BGRA for the decoded frame
    Codec stream_codec = Codec::RawBGRA;
    int w = 0, h = 0;

    // Client-side decode telemetry, accumulated per frame and reset on each server Stats message
    // (all touched only from the session thread, so no locking needed).
    double dec_ms_sum = 0.0;  // time spent turning payloads into displayable BGRA
    double dec_ms_sq_sum = 0.0; // sum of squared per-feed decode times, for the std-dev
    size_t dec_frames = 0;    // frames decoded in the current window
    size_t recv_bytes = 0;    // payload bytes received off the wire
    size_t out_bytes  = 0;    // decoded BGRA bytes produced
    std::vector<double> lat_ms; // end-to-end latency samples (stamp echo -> decoded frame)
    size_t lost = 0;          // frames lost on the unreliable (datagram) path this window

    void init()
    {
        // Prefer NVDEC (h264_cuvid) when this client has it and it opens (NVIDIA GPU + driver);
        // otherwise use the portable software decoder. Both decode the same standard H.264 stream
        // NVENC produces and output frames we convert to BGRA with libswscale.
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        if (codec)
        {
            ctx = avcodec_alloc_context3(codec);
            // The server encodes with max_b_frames = 0 and nvenc delay=0 (no reordering), but
            // cuvid still holds a 4-frame display-delay pipeline BY DEFAULT: the first decoded
            // frame only comes out after ~4 packets went in. At interactive rates that is just
            // added latency; at slow frame rates (huge N renders several seconds per frame) it
            // starved wait_for_geometry entirely -> "no stream received". delay=0 emits each
            // frame as soon as it is decoded.
            ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
            av_opt_set(ctx->priv_data, "delay", "0", 0);
            if (avcodec_open2(ctx, codec, nullptr) != 0)
            {
                avcodec_free_context(&ctx); // no usable NVDEC; fall back
                codec = nullptr;
            }
        }
        if (!codec)
        {
            codec = avcodec_find_decoder(AV_CODEC_ID_H264);
            ctx   = avcodec_alloc_context3(codec);
            ctx->flags |= AV_CODEC_FLAG_LOW_DELAY; // stream has no B-frames; decode 1-in-1-out
            avcodec_open2(ctx, codec, nullptr);
        }
        pkt   = av_packet_alloc();
        frame = av_frame_alloc();
        printf("rr-client: H.264 decoder = %s\n", codec ? codec->name : "(none)");
    }
    void set_geometry(const Hello& hello)
    {
        stream_codec = static_cast<Codec>(hello.codec);
        w = static_cast<int>(hello.width);
        h = static_cast<int>(hello.height);
        if (sws) { sws_freeContext(sws); sws = nullptr; }
        if (ctx) { avcodec_flush_buffers(ctx); } // re-sync on the next IDR
    }
    void feed(const uint8_t *payload, size_t len)
    {
        const auto t0 = std::chrono::steady_clock::now();
        recv_bytes += len;
        if (stream_codec == Codec::RawBGRA)
        {
            store_frame(payload, w, h);
            out_bytes += static_cast<size_t>(w) * h * 4;
            ++dec_frames;
            const double dt = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            dec_ms_sum += dt;
            dec_ms_sq_sum += dt * dt;
            return;
        }
        pkt->data = const_cast<uint8_t*>(payload);
        pkt->size = static_cast<int>(len);
        if (avcodec_send_packet(ctx, pkt) < 0) { return; }
        while (avcodec_receive_frame(ctx, frame) == 0)
        {
            int fw = frame->width, fh = frame->height; // decoded dims (authoritative)
            if (!sws)
            {
                sws = sws_getContext(fw, fh, static_cast<AVPixelFormat>(frame->format),
                    fw, fh, AV_PIX_FMT_BGRA, SWS_BILINEAR, nullptr, nullptr, nullptr);
            }
            tmp.resize(static_cast<size_t>(fw) * fh * 4);
            uint8_t *dst[4]   = { tmp.data(), nullptr, nullptr, nullptr };
            int dst_stride[4] = { fw * 4, 0, 0, 0 };
            sws_scale(sws, frame->data, frame->linesize, 0, fh, dst, dst_stride);
            store_frame(tmp.data(), fw, fh);
            out_bytes += static_cast<size_t>(fw) * fh * 4;
            ++dec_frames;
        }
        const double dt = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        dec_ms_sum += dt;
        dec_ms_sq_sum += dt * dt;
    }
    void destroy()
    {
        if (sws)   { sws_freeContext(sws); }
        if (frame) { av_frame_free(&frame); }
        if (pkt)   { av_packet_free(&pkt); }
        if (ctx)   { avcodec_free_context(&ctx); }
    }
};

// Handles one video-channel message (telemetry, geometry update, or a frame). echo_stamp is the
// server's echo of this client's newest input/heartbeat stamp (0 = none), valid on frame messages.
void feed_video(Decoder& dec, uint32_t flags, const uint8_t *payload, size_t len, uint32_t echo_stamp)
{
    if (flags & FRAME_STATS)
    {
        Stats st{};
        // Copy min(payload, sizeof): a newer client keeps the leading fields even if an older
        // server sends a shorter Stats (its appended encode_std_us then stays 0).
        std::memcpy(&st, payload, std::min(len, sizeof(st)));
        // fps/kbps/encode come from the server (encode_us/encode_std_us are ITS per-frame
        // production time); decode, end-to-end latency, loss and the size expansion are measured
        // here, over the frames since the last Stats.
        const double n       = dec.dec_frames > 0 ? static_cast<double>(dec.dec_frames) : 1.0;
        const double dec_ms  = dec.dec_ms_sum / n;
        const double dec_var = dec.dec_ms_sq_sum / n - dec_ms * dec_ms;
        const double dec_std = std::sqrt(std::max(0.0, dec_var));
        const double recv_kb = static_cast<double>(dec.recv_bytes) / n / 1000.0;
        const double out_kb  = static_cast<double>(dec.out_bytes) / n / 1000.0;
        // Latency percentiles + std-dev over this window's samples (heartbeats give ~20/s).
        double lat_mean = 0.0, lat_std = 0.0, lat_p50 = 0.0, lat_p95 = 0.0, lat_max = 0.0;
        if (!dec.lat_ms.empty())
        {
            std::sort(dec.lat_ms.begin(), dec.lat_ms.end());
            for (double v : dec.lat_ms) { lat_mean += v; }
            lat_mean /= static_cast<double>(dec.lat_ms.size());
            double sq = 0.0;
            for (double v : dec.lat_ms) { const double d = v - lat_mean; sq += d * d; }
            lat_std = std::sqrt(sq / static_cast<double>(dec.lat_ms.size()));
            lat_p50 = dec.lat_ms[dec.lat_ms.size() / 2];
            lat_p95 = dec.lat_ms[(dec.lat_ms.size() * 95) / 100];
            lat_max = dec.lat_ms.back();
        }
        const uint32_t ctrl = g.ctrl_sent.exchange(0);
        // LOD and denoise are distinct stages (server render_us includes both), so break them out and
        // show render as the pure draw/trace cost (render_us - lod - denoise); mirrors the on-screen HUD.
        // Pipeline order: lod | render | denoise (render shown pure = render_us - lod - denoise).
        char cons_render[96], cr_lod[24] = "", cr_dn[28] = "";
        if (st.lod_us > 5)     { snprintf(cr_lod, sizeof(cr_lod), "lod %.1f ms | ", st.lod_us / 1000.0); }
        if (st.denoise_us > 5) { snprintf(cr_dn, sizeof(cr_dn), " | denoise %.1f ms", st.denoise_us / 1000.0); }
        snprintf(cons_render, sizeof(cons_render), "%srender %.1f ms%s", cr_lod,
            std::max(0.0, (st.render_us - st.lod_us - st.denoise_us) / 1000.0), cr_dn);
        printf("%s[stats] %.1f fps, %u kbps | server %s %.2f+-%.2f ms | %s | decode %.2f+-%.2f ms | "
            "latency %.1f+-%.1f ms (p95 %.1f) | %zu lost | %.0f kB -> %.0f kB/frame (%.1fx larger)%s\n",
            ansi::dim(), st.fps_milli / 1000.0, st.kbps,
            dec.stream_codec == Codec::H264 ? "encode" : "readback",
            st.encode_us / 1000.0, st.encode_std_us / 1000.0, cons_render, dec_ms, dec_std,
            lat_mean, lat_std, lat_p95, dec.lost,
            recv_kb, out_kb, recv_kb > 0.0 ? out_kb / recv_kb : 0.0, ansi::rst());
        if (g_csv)
        {
            const char *phase = g_phase.load();
            if (phase[0] == '\0') { phase = ctrl > 0 ? "move" : "idle"; }
            fprintf(g_csv, "%.3f,%.1f,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%zu,%u,%s\n",
                now_ms() / 1000.0, st.fps_milli / 1000.0, st.kbps,
                st.encode_us / 1000.0, st.encode_std_us / 1000.0,
                st.compute_us / 1000.0, st.render_us / 1000.0, dec_ms, dec_std,
                lat_mean, lat_std, lat_p50, lat_p95, lat_max, dec.lost, ctrl, phase);
            fflush(g_csv);
        }
        dec.dec_ms_sum = 0.0; dec.dec_ms_sq_sum = 0.0; dec.dec_frames = 0;
        dec.recv_bytes = 0; dec.out_bytes = 0;
        dec.lat_ms.clear(); dec.lost = 0;
        {
            std::lock_guard<std::mutex> lock(g_hud.mtx);
            g_hud.fps        = st.fps_milli / 1000.0;
            g_hud.step       = st.step;
            g_hud.step_limit = st.step_limit;
            g_hud.particle_count     = st.particle_count;
            g_hud.particles_per_sec  = st.particles_per_sec;
            g_hud.compute_ms = st.compute_us / 1000.0; // server sim step time
            g_hud.sort_ms    = st.sort_us / 1000.0;    // server spatial-sort time (part of compute)
            g_hud.render_ms  = st.render_us / 1000.0;  // server GPU render time
            g_hud.lod_ms     = st.lod_us / 1000.0;     // server LOD reduction time (part of render)
            g_hud.denoise_ms = st.denoise_us / 1000.0; // server denoiser time (part of render)
            g_hud.encode_ms  = st.encode_us / 1000.0;  // server encode/readback time
            g_hud.decode_ms  = dec_ms;                 // client decode time (measured above)
            g_hud.vram_used_mb  = st.vram_used_mb;     // server GPU memory used/total (GPU HUD line)
            g_hud.vram_total_mb = st.vram_total_mb;
            g_hud.power_w       = st.power_w;          // server board power draw/limit (GPU HUD line)
            g_hud.power_limit_w = st.power_limit_w;
            g_hud.have_stats = true;
        }
        return;
    }
    if (flags & FRAME_HELLO)
    {
        // Dormant on the server by default, but handled for completeness: geometry changed.
        if (len >= sizeof(Hello))
        {
            Hello h{}; std::memcpy(&h, payload, sizeof(h));
            dec.set_geometry(h);
            hud_set_server(h, g_transport);
        }
        return;
    }
    dec.feed(payload, len);
    // End-to-end latency: client stamp -> (server queue+render+encode) -> network -> decoded
    // frame, all on this client's clock (the stamp merely round-tripped).
    if (echo_stamp != 0)
    {
        const double v = static_cast<double>(now_ms() - echo_stamp);
        dec.lat_ms.push_back(v);
        // Lightly smoothed for the HUD (samples arrive ~20/s; raw values flicker unreadably).
        std::lock_guard<std::mutex> lock(g_hud.mtx);
        g_hud.latency_ms = g_hud.latency_ms < 0.0 ? v : 0.8 * g_hud.latency_ms + 0.2 * v;
    }
}

void fill_auth(AuthMsg& a, const std::string& token)
{
    a.magic = AUTH_MAGIC;
    size_t n = token.size() < TOKEN_MAX ? token.size() : static_cast<size_t>(TOKEN_MAX);
    std::memcpy(a.token, token.data(), n);
    // Announce our hostname so the server can tag its benchmark CSV with this client.
    char host[HOST_MAX]{};
    if (gethostname(host, sizeof(host) - 1) == 0) { std::memcpy(a.client, host, sizeof(host)); }
}

// ============================================================================================
// TCP session
// ============================================================================================
bool tcpSendAll(int fd, const void *buf, size_t len)
{
    auto *p = static_cast<const char*>(buf);
    for (size_t s = 0; s < len; )
    {
        ssize_t k = send(fd, p + s, len - s, MSG_NOSIGNAL);
        if (k <= 0) { return false; }
        s += static_cast<size_t>(k);
    }
    return true;
}
bool tcpRecvAll(int fd, void *buf, size_t len)
{
    auto *p = static_cast<char*>(buf);
    for (size_t r = 0; r < len; )
    {
        ssize_t k = recv(fd, p + r, len - r, 0);
        if (k <= 0) { return false; }
        r += static_cast<size_t>(k);
    }
    return true;
}

// Returns true if a TCP session was established and ran.
bool run_tcp(const char *host, const char *port, const std::string& token, Decoder& dec)
{
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host, port, &hints, &res) != 0) { return false; }
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) != 0)
    {
        if (fd >= 0) { close(fd); }
        freeaddrinfo(res);
        return false;
    }
    freeaddrinfo(res);
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    AuthMsg auth{}; fill_auth(auth, token);
    if (!tcpSendAll(fd, &auth, sizeof(auth))) { close(fd); return false; }

    Hello hello{};
    if (!tcpRecvAll(fd, &hello, sizeof(hello)) || hello.magic != PROTOCOL_MAGIC)
    {
        fprintf(stderr, "%sTCP: invalid server hello (rejected? wrong token?)%s\n", ansi::red(), ansi::rst());
        close(fd); return false;
    }
    dec.set_geometry(hello);
    g_transport = "TCP";
    hud_set_server(hello, g_transport);
    bench_csv_open(hello);
    printf("%sconnected over TCP: %ux%u (%s)%s%.*s%s\n", ansi::grn(), hello.width, hello.height,
        static_cast<Codec>(hello.codec) == Codec::H264 ? "H.264" : "raw",
        hello.gpu[0] ? " on " : "", GPU_MAX, hello.gpu, ansi::rst());

    std::vector<uint8_t> payload;
    uint64_t last_hb = 0;
    while (!g.quit.load())
    {
        // Poll so the loop wakes to drain UI control even between frames.
        pollfd pfd{ fd, POLLIN, 0 };
        int pr = poll(&pfd, 1, 8);
        if (pr < 0) { if (errno == EINTR) { continue; } break; }
        if (pfd.revents & POLLIN)
        {
            FrameHeader fh{};
            if (!tcpRecvAll(fd, &fh, sizeof(fh))) { break; }
            payload.resize(fh.size);
            if (fh.size && !tcpRecvAll(fd, payload.data(), fh.size)) { break; }
            feed_video(dec, fh.flags, payload.data(), fh.size, fh.echo_stamp);
        }
        maybe_heartbeat(last_hb); // keep latency sampled even without interaction
        // Send any queued control events.
        std::deque<ControlMsg> batch;
        { std::lock_guard<std::mutex> lock(g.ctrl_mtx); batch.swap(g.outgoing); }
        for (const auto& m : batch) { if (!tcpSendAll(fd, &m, sizeof(m))) { g.quit.store(true); break; } }
    }
    close(fd);
    return true;
}

#ifdef MIMIR_RRC_HAVE_QUIC
// ============================================================================================
// QUIC session (ngtcp2 + OpenSSL crypto binding)
// ============================================================================================
struct Quic
{
    ngtcp2_crypto_conn_ref conn_ref{};
    int fd = -1;
    sockaddr_storage local_addr{};
    socklen_t local_addrlen = sizeof(local_addr);
    sockaddr_storage remote_addr{};
    socklen_t remote_addrlen = 0;
    SSL_CTX *ssl_ctx = nullptr;
    SSL *ssl = nullptr;
    ngtcp2_crypto_ossl_ctx *ossl_ctx = nullptr;
    ngtcp2_conn *conn = nullptr;
    ngtcp2_ccerr last_error{};
    bool handshake_done = false;
    int64_t control_stream = -1;
    std::vector<uint8_t> vbuf;     // received video bytes
    bool got_hello = false;
    std::vector<uint8_t> ctrl_out; // bytes queued for the control stream
    size_t ctrl_off = 0;
    Decoder *dec = nullptr;

    // Unreliable video: reassembly of DatagramFrag-headed QUIC datagrams into whole frames.
    // Fragments of one frame arrive (possibly reordered) between other frames' fragments only
    // on loss; normally a frame completes before the next begins.
    bool     rf_active = false;    // currently reassembling rf_id
    uint32_t rf_id = 0;
    std::vector<uint8_t> rf_buf;
    size_t   rf_received = 0;      // distinct payload bytes received (fragments are disjoint)
    uint32_t rf_flags = 0;
    uint32_t rf_echo = 0;
    uint32_t next_frame_id = 0;    // first id not yet completed (gap => frames lost)
    bool     need_keyframe = false;// lost a frame: discard P-frames until the next IDR
    uint64_t last_kf_req = 0;      // rate-limits RequestKeyframe to one per 200 ms

    // A frame was lost (never completed / skipped). P-frames after the gap reference data this
    // client never got, so ask for an IDR and drop everything until it arrives.
    void onLoss(size_t count)
    {
        dec->lost += count;
        need_keyframe = true;
        requestKeyframe();
    }
    void requestKeyframe()
    {
        const uint64_t now = now_ns();
        if (now - last_kf_req >= 200ull * 1000000ull)
        {
            ui_control(ControlKind::RequestKeyframe);
            last_kf_req = now;
        }
    }
};

ngtcp2_conn* quic_get_conn(ngtcp2_crypto_conn_ref *ref)
{ return static_cast<Quic*>(ref->user_data)->conn; }

void quic_rand(uint8_t *dest, size_t destlen, const ngtcp2_rand_ctx*)
{ RAND_bytes(dest, static_cast<int>(destlen)); }

#if MIMIR_NGTCP2_CALLBACKS2
int quic_get_new_cid(ngtcp2_conn*, ngtcp2_cid *cid, ngtcp2_stateless_reset_token *token,
    size_t cidlen, void*)
{
    if (RAND_bytes(cid->data, static_cast<int>(cidlen)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    cid->datalen = cidlen;
    if (RAND_bytes(token->data, sizeof(token->data)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    return 0;
}
#else // ngtcp2 < 1.22.0: token is a raw uint8_t buffer, not a struct
int quic_get_new_cid(ngtcp2_conn*, ngtcp2_cid *cid, uint8_t *token, size_t cidlen, void*)
{
    if (RAND_bytes(cid->data, static_cast<int>(cidlen)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    cid->datalen = cidlen;
    if (RAND_bytes(token, NGTCP2_STATELESS_RESET_TOKENLEN) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    return 0;
}
#endif

int quic_handshake_done(ngtcp2_conn*, void *user_data)
{ static_cast<Quic*>(user_data)->handshake_done = true; return 0; }

int quic_recv_stream_data(ngtcp2_conn *conn, uint32_t, int64_t stream_id, uint64_t,
    const uint8_t *data, size_t datalen, void *user_data, void*)
{
    auto *q = static_cast<Quic*>(user_data);
    q->vbuf.insert(q->vbuf.end(), data, data + datalen);
    ngtcp2_conn_extend_max_stream_offset(conn, stream_id, datalen);
    ngtcp2_conn_extend_max_offset(conn, datalen);
    return 0;
}

// One unreliable video datagram: a DatagramFrag header + fragment payload. Reassembles whole
// frames; anything lost is never retransmitted, so incomplete frames are dropped and decoding
// resumes at the next keyframe.
int quic_recv_datagram(ngtcp2_conn*, uint32_t, const uint8_t *data, size_t datalen, void *user_data)
{
    auto *q = static_cast<Quic*>(user_data);
    if (datalen < sizeof(DatagramFrag)) { return 0; }
    DatagramFrag h{};
    std::memcpy(&h, data, sizeof(h));
    const uint8_t *payload = data + sizeof(h);
    const size_t plen = datalen - sizeof(h);
    if (h.frame_bytes == 0 || plen == 0
        || static_cast<size_t>(h.offset) + plen > h.frame_bytes) { return 0; } // malformed
    if (q->rf_active && h.frame_id != q->rf_id)
    {
        if (h.frame_id < q->rf_id) { return 0; } // stale straggler of an already-abandoned frame
        q->onLoss(1);                            // a newer frame began: the rest of rf_id is gone
        q->rf_active = false;
    }
    if (!q->rf_active)
    {
        if (h.frame_id < q->next_frame_id) { return 0; }          // duplicate/very late fragment
        if (h.frame_id > q->next_frame_id)                        // ids skipped entirely
        {
            q->onLoss(h.frame_id - q->next_frame_id);
            q->next_frame_id = h.frame_id;
        }
        q->rf_active = true;
        q->rf_id = h.frame_id;
        q->rf_buf.assign(h.frame_bytes, 0);
        q->rf_received = 0;
        q->rf_flags = h.flags;
        q->rf_echo = h.echo_stamp;
    }
    std::memcpy(q->rf_buf.data() + h.offset, payload, plen);
    q->rf_received += plen;
    if (q->rf_received >= q->rf_buf.size())
    {
        q->rf_active = false;
        q->next_frame_id = q->rf_id + 1;
        if (q->rf_flags & FRAME_KEYFRAME) { q->need_keyframe = false; }
        if (q->need_keyframe) { q->requestKeyframe(); return 0; } // undecodable until an IDR
        feed_video(*q->dec, q->rf_flags, q->rf_buf.data(), q->rf_buf.size(), q->rf_echo);
    }
    return 0;
}

void quic_process_video(Quic *q)
{
    size_t pos = 0;
    if (!q->got_hello)
    {
        if (q->vbuf.size() < sizeof(Hello)) { return; }
        Hello hello{};
        std::memcpy(&hello, q->vbuf.data(), sizeof(Hello));
        if (hello.magic != PROTOCOL_MAGIC) { fprintf(stderr, "QUIC: bad hello\n"); g.quit.store(true); return; }
        q->got_hello = true;
        pos = sizeof(Hello);
        q->dec->set_geometry(hello);
        g_transport = "QUIC";
        hud_set_server(hello, g_transport);
        bench_csv_open(hello);
        printf("%sconnected over QUIC: %ux%u (%s)%s%.*s%s\n", ansi::grn(), hello.width, hello.height,
            static_cast<Codec>(hello.codec) == Codec::H264 ? "H.264" : "raw",
            hello.gpu[0] ? " on " : "", GPU_MAX, hello.gpu, ansi::rst());
    }
    for (;;)
    {
        if (q->vbuf.size() - pos < sizeof(FrameHeader)) { break; }
        FrameHeader fh{};
        std::memcpy(&fh, q->vbuf.data() + pos, sizeof(FrameHeader));
        if (q->vbuf.size() - pos - sizeof(FrameHeader) < fh.size) { break; }
        feed_video(*q->dec, fh.flags, q->vbuf.data() + pos + sizeof(FrameHeader), fh.size, fh.echo_stamp);
        pos += sizeof(FrameHeader) + fh.size;
    }
    if (pos > 0) { q->vbuf.erase(q->vbuf.begin(), q->vbuf.begin() + static_cast<long>(pos)); }
}

bool quic_send_packet(Quic *q, const uint8_t *data, size_t len)
{
    for (;;)
    {
        ssize_t n = send(q->fd, data, len, 0);
        if (n < 0 && errno == EINTR) { continue; }
        return n >= 0;
    }
}

bool quic_pump_read(Quic *q)
{
    uint8_t buf[65536];
    for (;;)
    {
        ssize_t n = recv(q->fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n < 0) { return (errno == EAGAIN || errno == EWOULDBLOCK); }
        if (n == 0) { return true; }
        ngtcp2_path path{};
        path.local.addr = reinterpret_cast<sockaddr*>(&q->local_addr);
        path.local.addrlen = q->local_addrlen;
        path.remote.addr = reinterpret_cast<sockaddr*>(&q->remote_addr);
        path.remote.addrlen = q->remote_addrlen;
        ngtcp2_pkt_info pi{};
        if (ngtcp2_conn_read_pkt(q->conn, &path, &pi, buf, static_cast<size_t>(n), now_ns()) != 0)
        {
            return false;
        }
    }
}

bool quic_pump_write(Quic *q)
{
    // Move queued UI control into the control-stream byte buffer.
    {
        std::deque<ControlMsg> batch;
        { std::lock_guard<std::mutex> lock(g.ctrl_mtx); batch.swap(g.outgoing); }
        for (const auto& m : batch)
        {
            auto *p = reinterpret_cast<const uint8_t*>(&m);
            q->ctrl_out.insert(q->ctrl_out.end(), p, p + sizeof(m));
        }
    }
    uint8_t buf[1452];
    ngtcp2_path_storage ps; ngtcp2_path_storage_zero(&ps);
    ngtcp2_pkt_info pi{};
    const ngtcp2_tstamp ts = now_ns();
    for (;;)
    {
        if (q->control_stream == -1 && q->handshake_done)
        {
            int64_t sid = -1;
            if (ngtcp2_conn_open_uni_stream(q->conn, &sid, nullptr) == 0) { q->control_stream = sid; }
        }
        int64_t stream_id = -1;
        ngtcp2_vec vec{};
        size_t vcnt = 0;
        if (q->control_stream != -1 && q->ctrl_off < q->ctrl_out.size())
        {
            stream_id = q->control_stream;
            vec.base = q->ctrl_out.data() + q->ctrl_off;
            vec.len = q->ctrl_out.size() - q->ctrl_off;
            vcnt = 1;
        }
        ngtcp2_ssize wdatalen = 0;
        ngtcp2_ssize nwrite = ngtcp2_conn_writev_stream(q->conn, &ps.path, &pi, buf, sizeof(buf),
            &wdatalen, NGTCP2_WRITE_STREAM_FLAG_MORE, stream_id, vcnt ? &vec : nullptr, vcnt, ts);
        if (nwrite == NGTCP2_ERR_WRITE_MORE) { if (wdatalen > 0) { q->ctrl_off += static_cast<size_t>(wdatalen); } continue; }
        if (nwrite < 0) { return false; }
        if (wdatalen > 0) { q->ctrl_off += static_cast<size_t>(wdatalen); }
        if (q->ctrl_off > 0 && q->ctrl_off == q->ctrl_out.size()) { q->ctrl_out.clear(); q->ctrl_off = 0; }
        if (nwrite == 0) { return true; }
        if (!quic_send_packet(q, buf, static_cast<size_t>(nwrite))) { return false; }
    }
}

void quic_free(Quic *q)
{
    if (q->conn)     { ngtcp2_conn_del(q->conn); }
    if (q->ossl_ctx) { ngtcp2_crypto_ossl_ctx_del(q->ossl_ctx); }
    if (q->ssl)      { SSL_set_app_data(q->ssl, nullptr); SSL_free(q->ssl); }
    if (q->ssl_ctx)  { SSL_CTX_free(q->ssl_ctx); }
    if (q->fd >= 0)  { close(q->fd); }
}

// Returns true if a QUIC session was established (so the caller need not fall back).
bool run_quic(const char *host, const char *port, const std::string& token, Decoder& dec)
{
    Quic q;
    q.dec = &dec;
    ngtcp2_ccerr_default(&q.last_error);

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host, port, &hints, &res) != 0) { return false; }
    q.fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (q.fd < 0 || connect(q.fd, res->ai_addr, res->ai_addrlen) != 0)
    {
        if (q.fd >= 0) { close(q.fd); }
        freeaddrinfo(res);
        return false;
    }
    std::memcpy(&q.remote_addr, res->ai_addr, res->ai_addrlen);
    q.remote_addrlen = res->ai_addrlen;
    freeaddrinfo(res);
    if (getsockname(q.fd, reinterpret_cast<sockaddr*>(&q.local_addr), &q.local_addrlen) != 0)
    {
        close(q.fd); return false;
    }

    if (ngtcp2_crypto_ossl_init() != 0) { close(q.fd); return false; }
    q.ssl_ctx = SSL_CTX_new(TLS_client_method());
    if (!q.ssl_ctx) { close(q.fd); return false; }
    SSL_CTX_set_min_proto_version(q.ssl_ctx, TLS1_3_VERSION);
    SSL_CTX_set_max_proto_version(q.ssl_ctx, TLS1_3_VERSION);
    q.ssl = SSL_new(q.ssl_ctx);
    if (!q.ssl || ngtcp2_crypto_ossl_ctx_new(&q.ossl_ctx, q.ssl) != 0
        || ngtcp2_crypto_ossl_configure_client_session(q.ssl) != 0)
    {
        quic_free(&q); return false;
    }
    q.conn_ref.get_conn = quic_get_conn;
    q.conn_ref.user_data = &q;
    SSL_set_app_data(q.ssl, &q.conn_ref);
    SSL_set_connect_state(q.ssl);
    SSL_set_alpn_protos(q.ssl, ALPN, sizeof(ALPN));

    ngtcp2_callbacks cb{};
    cb.client_initial           = ngtcp2_crypto_client_initial_cb;
    cb.recv_crypto_data         = ngtcp2_crypto_recv_crypto_data_cb;
    cb.encrypt                  = ngtcp2_crypto_encrypt_cb;
    cb.decrypt                  = ngtcp2_crypto_decrypt_cb;
    cb.hp_mask                  = ngtcp2_crypto_hp_mask_cb;
    cb.recv_retry               = ngtcp2_crypto_recv_retry_cb;
    cb.update_key               = ngtcp2_crypto_update_key_cb;
    cb.delete_crypto_aead_ctx   = ngtcp2_crypto_delete_crypto_aead_ctx_cb;
    cb.delete_crypto_cipher_ctx = ngtcp2_crypto_delete_crypto_cipher_ctx_cb;
    cb.version_negotiation      = ngtcp2_crypto_version_negotiation_cb;
    cb.rand                     = quic_rand;
    cb.handshake_completed      = quic_handshake_done;
#if MIMIR_NGTCP2_CALLBACKS2
    cb.get_path_challenge_data2 = ngtcp2_crypto_get_path_challenge_data2_cb;
    cb.get_new_connection_id2   = quic_get_new_cid;
#else // ngtcp2 < 1.22.0: the callbacks without the "2" suffix
    cb.get_path_challenge_data  = ngtcp2_crypto_get_path_challenge_data_cb;
    cb.get_new_connection_id    = quic_get_new_cid;
#endif
    cb.recv_stream_data         = quic_recv_stream_data;
    cb.recv_datagram            = quic_recv_datagram;

    ngtcp2_cid dcid{}, scid{};
    dcid.datalen = NGTCP2_MIN_INITIAL_DCIDLEN;
    RAND_bytes(dcid.data, static_cast<int>(dcid.datalen));
    scid.datalen = 8;
    RAND_bytes(scid.data, static_cast<int>(scid.datalen));

    ngtcp2_path path{};
    path.local.addr = reinterpret_cast<sockaddr*>(&q.local_addr);
    path.local.addrlen = q.local_addrlen;
    path.remote.addr = reinterpret_cast<sockaddr*>(&q.remote_addr);
    path.remote.addrlen = q.remote_addrlen;

    ngtcp2_settings settings{};
    ngtcp2_settings_default(&settings);
    settings.initial_ts = now_ns();
    ngtcp2_transport_params params{};
    ngtcp2_transport_params_default(&params);
    params.initial_max_streams_uni     = 3;
    params.initial_max_stream_data_uni = 16 * 1024 * 1024;
    params.initial_max_data            = 64 * 1024 * 1024;
    params.max_datagram_frame_size     = 65535; // advertise RFC 9221: unreliable video frames

    if (ngtcp2_conn_client_new(&q.conn, &dcid, &scid, &path, NGTCP2_PROTO_VER_V1,
            &cb, &settings, &params, nullptr, &q) != 0)
    {
        quic_free(&q); return false;
    }
    ngtcp2_conn_set_tls_native_handle(q.conn, q.ossl_ctx);

    // Queue auth first so it leads the control stream; the server withholds video until it checks.
    AuthMsg auth{}; fill_auth(auth, token);
    { auto *p = reinterpret_cast<const uint8_t*>(&auth); q.ctrl_out.insert(q.ctrl_out.end(), p, p + sizeof(auth)); }
    if (!quic_pump_write(&q)) { quic_free(&q); return false; }

    const uint64_t t0 = now_ns();
    uint64_t last_hb = 0;
    while (!g.quit.load())
    {
        if (q.handshake_done) { maybe_heartbeat(last_hb); } // latency sampling heartbeat
        ngtcp2_tstamp expiry = ngtcp2_conn_get_expiry(q.conn);
        ngtcp2_tstamp now = now_ns();
        int timeout = 16;
        if (expiry != UINT64_MAX)
        {
            int e = (expiry <= now) ? 0 : static_cast<int>((expiry - now) / NGTCP2_MILLISECONDS);
            if (e < timeout) { timeout = e; }
        }
        pollfd pfd{ q.fd, POLLIN, 0 };
        int pr = poll(&pfd, 1, timeout);
        if (pr < 0) { if (errno == EINTR) { continue; } break; }
        if ((pfd.revents & POLLIN) && !quic_pump_read(&q)) { break; }
        quic_process_video(&q);
        if (ngtcp2_conn_get_expiry(q.conn) <= now_ns())
        {
            if (ngtcp2_conn_handle_expiry(q.conn, now_ns()) != 0) { break; }
        }
        if (!quic_pump_write(&q)) { break; }
        if (!q.handshake_done && (now_ns() - t0) > 3ull * NGTCP2_SECONDS)
        {
            fprintf(stderr, "QUIC handshake timed out\n");
            break;
        }
    }
    bool established = q.got_hello;
    quic_free(&q);
    return established;
}
#endif // MIMIR_RRC_HAVE_QUIC

// ============================================================================================
// Session thread: pick transport, run until shutdown.
// ============================================================================================
void session_thread(std::string host, std::string port, std::string token, std::string mode)
{
    Decoder dec;
    dec.init();
#ifdef MIMIR_RRC_HAVE_QUIC
    if (mode == "tcp")
    {
        run_tcp(host.c_str(), port.c_str(), token, dec);
    }
    else
    {
        bool ok = run_quic(host.c_str(), port.c_str(), token, dec);
        if (!ok && mode != "quic")
        {
            printf("QUIC unavailable; falling back to TCP\n");
            run_tcp(host.c_str(), port.c_str(), token, dec);
        }
    }
#else
    // TCP-only build: no ngtcp2 was available at compile time. "auto" already means TCP here;
    // an explicit "quic" request cannot be honoured, so note it and use TCP.
    if (mode == "quic")
    {
        fprintf(stderr, "rr-client: built without QUIC support; using TCP\n");
    }
    run_tcp(host.c_str(), port.c_str(), token, dec);
#endif // MIMIR_RRC_HAVE_QUIC
    dec.destroy();
    g.running.store(false);
}

// ============================================================================================
// Window (interactive) and headless (test) front-ends.
// ============================================================================================
// Mouse/keyboard capture lives in the shared rr_input.hpp helper; `g_client_input` is wired to the
// transport (emit) and the client-UI hooks in main().
mimir::rr::InputCapture g_client_input;

// Waits (up to g_first_frame_timeout_sec) for the session to deliver the first frame so we know the
// stream geometry. Generous on purpose: at huge particle counts the server's first path-traced frame
// can take minutes, and a dead connection already exits early via g.running.
bool wait_for_geometry(int& w, int& h)
{
    const int max_iters = g_first_frame_timeout_sec * 100; // 10 ms per iter => 100 iters/second
    for (int i = 0; i < max_iters && g.running.load(); ++i)
    {
        { std::lock_guard<std::mutex> lock(g.frame_mtx); if (g.have_geometry) { w = g.w; h = g.h; return true; } }
        if (i > 0 && i % 500 == 0) { printf("rr-client: waiting for first frame (%d s)...\n", i / 100); }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

int run_window()
{
    int w = 0, h = 0;
    if (!wait_for_geometry(w, h)) { fprintf(stderr, "%sno stream received%s\n", ansi::red(), ansi::rst()); return EXIT_FAILURE; }

    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return EXIT_FAILURE; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // window is freely resizable; the frame is stretched
    // Open at the requested window size, or the stream resolution if none was given. The frame is
    // stretched to fill whatever the window is, so this never touches the server-side resolution.
    const int win_w = (g_win_w > 0) ? g_win_w : w;
    const int win_h = (g_win_h > 0) ? g_win_h : h;
    GLFWwindow *window = glfwCreateWindow(win_w, win_h, "mimir rr-client", nullptr, nullptr);
    if (!window) { fprintf(stderr, "window creation failed\n"); glfwTerminate(); return EXIT_FAILURE; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    // Wire the shared input capture to this client's transport (emit) and UI, then install it.
    g_client_input.emit          = [](ControlKind k, float a, float b) { ui_control(k, a, b); };
    g_client_input.is_fly        = [] { return g_fly.load(std::memory_order_relaxed); };
    g_client_input.on_toggle_hud = [] { g_hud.visible.store(!g_hud.visible.load()); };
    g_client_input.on_quit       = [window] { glfwSetWindowShouldClose(window, GLFW_TRUE); };
    g_client_input.install(window);
    hud_font_init(); // load the anti-aliased HUD font (bitmap fallback if none found)

    std::vector<unsigned char> display;
    uint64_t shown_seq = 0;
    int cur_w = w, cur_h = h;
    printf(g_fly.load() ? "camera: fly (left-drag look, WASD move)\n"
                        : "camera: trackball (left-drag orbit, right-drag zoom, middle-drag pan)\n");
    while (!glfwWindowShouldClose(window) && g.running.load())
    {
        glfwPollEvents();
        // Held WASD -> continuous CameraMove (fly only; the helper no-ops for trackball servers).
        g_client_input.pollMovement(window);
        {
            std::lock_guard<std::mutex> lock(g.frame_mtx);
            if (g.frame_seq != shown_seq)
            {
                display = g.latest;
                cur_w = g.w; cur_h = g.h;
                shown_seq = g.frame_seq;
            }
        }
        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT);
        if (!display.empty() && cur_w > 0 && cur_h > 0)
        {
            // Stretch the cur_w x cur_h frame to fill the (resizable) window framebuffer; the
            // vertical flip handles Vulkan's top-row-first layout.
            glPixelZoom(static_cast<float>(fbw) / cur_w, -static_cast<float>(fbh) / cur_h);
            glRasterPos2f(-1.f, 1.f);
            glDrawPixels(cur_w, cur_h, GL_BGRA, GL_UNSIGNED_BYTE, display.data());
        }
        if (g_hud.visible.load()) { hud_draw(fbh); }
        glfwSwapBuffers(window);
    }

    g.quit.store(true);
    ui_control(ControlKind::Quit);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}

void savePpm(const char *path, const std::vector<unsigned char>& bgra, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i)
    {
        f.put(static_cast<char>(bgra[i * 4 + 2]));
        f.put(static_cast<char>(bgra[i * 4 + 1]));
        f.put(static_cast<char>(bgra[i * 4 + 0]));
    }
    printf("saved %s (%dx%d)\n", path, w, h);
}

int run_headless(int frames)
{
    // Receive 'frames' frames (with the sim left running), save the last one, then quit.
    // A --benchmark script ending sets g.quit and stops the wait early.
    while (g.running.load() && !g.quit.load() && g.frames.load() < static_cast<uint64_t>(frames))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    int rc = EXIT_FAILURE;
    {
        std::lock_guard<std::mutex> lock(g.frame_mtx);
        if (g.have_geometry && !g.latest.empty())
        {
            savePpm("rr-client.ppm", g.latest, g.w, g.h);
            rc = EXIT_SUCCESS;
        }
    }
    printf("received %llu frames\n", static_cast<unsigned long long>(g.frames.load()));
    g.quit.store(true);
    ui_control(ControlKind::Quit);
    return rc;
}

// ============================================================================================
// --benchmark: deterministic scripted camera, so runs against different servers are comparable.
// ============================================================================================
// Emits the same control stream every run at a steady 60 Hz "mouse" cadence, in five phases of
// equal length (12 s each, 60 s total) spanning a static-to-high-motion gradient: far (static
// baseline outside the cloud), zoom-scale orbit, dive in, look around from within (peak motion),
// then hold still inside. Magnitudes are tuned to the engine's default camera (LookAt at z=-2.85
// over a roughly unit-sized cloud): the zoom leg dives ~3.2 world units in, deep enough to sit
// among the phenomena (but short of clipping into the spheres). Phases are published to g_phase so
// every CSV row is labeled for plot shading.
void bench_thread()
{
    // Wait for the stream to come up so the script timeline starts at the first frame (same
    // generous window as wait_for_geometry: the first frame can take minutes at huge N).
    const int max_iters = g_first_frame_timeout_sec * 100; // 10 ms per iter => 100 iters/second
    for (int i = 0; i < max_iters && g.running.load() && !g.quit.load(); ++i)
    {
        { std::lock_guard<std::mutex> lock(g.frame_mtx); if (g.have_geometry) { break; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // The five phase tokens (far/orbit/zoom_in/look_around/inside) map to the plot's "Client
    // Camera" legend, each tagged with its motion class (see research/scripts/plot_benchmark.py).
    // The orbit magnitude scales inversely with duration to keep one full orbit; the zoom leg is
    // deliberately faster than that so it ends deep among the phenomena rather than at their edge.
    struct Step { const char *phase; double secs; ControlKind kind; float a, b; };
    static const Step script[] = {
        { "far",     12.0,  ControlKind::None,         0.0f,   0.f }, // static baseline, outside the cloud
        { "orbit",   12.0,  ControlKind::CameraRotate, 0.50f,  0.f }, // ~360 deg yaw around the cloud
        { "zoom_in", 12.0,  ControlKind::CameraZoom,   0.90f,  0.f }, // dive in ~3.2 units, deep inside
        // Look around from within: turn the gaze in place (CameraLook = eye fixed), not another
        // orbit. Symmetric yaw then pitch sweeps that each return to center. 12 s total.
        { "look_around", 2.25, ControlKind::CameraLook,  1.0f,  0.f }, // yaw to one side
        { "look_around", 4.50, ControlKind::CameraLook, -1.0f,  0.f }, //  ... across to the other
        { "look_around", 2.25, ControlKind::CameraLook,  1.0f,  0.f }, //  ... back to center
        { "look_around", 0.75, ControlKind::CameraLook,  0.0f,  1.5f }, // tilt up
        { "look_around", 1.50, ControlKind::CameraLook,  0.0f, -1.5f }, //  ... down past center
        { "look_around", 0.75, ControlKind::CameraLook,  0.0f,  1.5f }, //  ... back to center
        { "inside",  12.0,  ControlKind::None,         0.0f,   0.f }, // static, now within the dense cloud
    };

    printf("benchmark: scripted camera starting "
        "(60 s: far/orbit/zoom_in/look_around/inside, 12 s each)\n");
    for (const auto& s : script)
    {
        if (!g.running.load() || g.quit.load()) { break; }
        g_phase.store(s.phase);
        printf("benchmark: phase %-11s (%.2g s)\n", s.phase, s.secs);
        const int ticks = static_cast<int>(s.secs * 60.0);
        for (int t = 0; t < ticks && g.running.load() && !g.quit.load(); ++t)
        {
            if (s.kind != ControlKind::None) { ui_control(s.kind, s.a, s.b); }
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    g_phase.store("done");
    printf("benchmark: script complete\n");
    g.quit.store(true);
    ui_control(ControlKind::Quit);
}

} // namespace

static void usage(const char *prog)
{
    printf(
        "Usage: %s [host] [port] [token] [transport] [frames] [--window W H] [--benchmark out.csv]\n"
        "\n"
        "  host       Server hostname or IP address       (default: 127.0.0.1)\n"
        "  port       Server port                         (default: 9000)\n"
        "  token      Shared secret (must match server;   (default: empty)\n"
        "             leave empty if server has none)\n"
        "  transport  auto (default) | quic | tcp\n"
        "             auto: tries QUIC first, falls back to TCP\n"
        "             tcp:  required when connecting through an ssh -L tunnel\n"
        "             quic: H.264 frames ride unreliable QUIC datagrams (lost frames are\n"
        "                   skipped, never retransmitted; decode resumes at a keyframe)\n"
        "  frames     If > 0, run headless: receive N frames, save rr-client.ppm, exit\n"
        "             (default: 0 = open interactive window)\n"
        "\n"
        "Window keys / mouse:\n"
        "  Left-drag  Trackball server: orbit the scene. Fly server (rr-server --fly): look around.\n"
        "  Right/Mid  Trackball server: right-drag zoom, middle-drag pan (no effect on a Fly server).\n"
        "  W A S D    Fly server only: move (forward follows the gaze, so look up + W climbs).\n"
        "             The camera model is chosen by the server and detected automatically.\n"
        "  H          Toggle the HUD overlay (server user@host:port, transport/codec,\n"
        "             latency, fps, simulation progress)\n"
        "  P          Pause/resume the simulation\n"
        "  Q / Esc    Quit\n"
        "\n"
        "Flags (order-independent):\n"
        "  --window W H   Open the viewer window at W by H (e.g. --window 1280 720), matching the\n"
        "                 server's 'width height' order. Default: the stream resolution announced\n"
        "                 by the server. Purely local -- the frame is stretched to fill the\n"
        "                 window; the server keeps rendering at its own resolution (not\n"
        "                 renegotiated).\n"
        "  --first-frame-timeout SECONDS\n"
        "                 How long to wait for the server's first frame before giving up with\n"
        "                 'no stream received' (default: 300). The first path-traced frame is a\n"
        "                 cold full build + trace and scales with particle count: ~62 s at 2^29,\n"
        "                 several minutes at 2^30. Raise this when driving very large scenes.\n"
        "  --benchmark P  Drive the camera with a deterministic 60 s script in five 12 s phases\n"
        "                 spanning a static-to-high-motion gradient (far: static baseline; orbit:\n"
        "                 one full orbit; zoom_in: dive into the cloud; look_around: turn the gaze\n"
        "                 in place from within, peak motion; inside: hold still within the cloud),\n"
        "                 quit, and write the per-second telemetry time series to an auto-named CSV.\n"
        "                 P is a path+prefix; the full name is assembled once connected as\n"
        "                   <P>-<YYYYMMDD>-rr-client-c<client>-s<server>-<gpu>.csv\n"
        "                 so it pairs with the server's file. Columns: time_s,fps,kbps,server_ms,\n"
        "                 server_ms_std,decode_ms,decode_ms_std,lat_mean_ms,lat_std_ms,lat_p50_ms,\n"
        "                 lat_p95_ms,lat_max_ms,lost,ctrl_events,phase (the *_std columns feed the\n"
        "                 plot's error bands; 'phase' labels the script phases for shading). The\n"
        "                 control stream is identical every run, so results from different servers\n"
        "                 are directly comparable. Pair with the server's --benchmark prefix.\n"
        "                 Windowed by default; give a large frames value (e.g. 99999) to run\n"
        "                 headless — the script's end stops the run.\n"
        "\n"
        "All other arguments are positional and optional; omitted trailing args use their defaults.\n"
        "\n"
        "Examples:\n"
        "  # Connect to a local server (or via ssh -L 9000:localhost:9000):\n"
        "  %s\n"
        "\n"
        "  # Connect to a remote server by IP over TCP (e.g. through ssh tunnel):\n"
        "  %s 127.0.0.1 9000 \"\" tcp\n"
        "\n"
        "  # Connect to a server with a shared secret, transport auto-selected:\n"
        "  %s 192.168.1.10 9000 mysecret\n"
        "\n"
        "  # Headless test: receive 10 frames, save rr-client.ppm, then exit:\n"
        "  %s 127.0.0.1 9000 \"\" tcp 10\n"
        "\n"
        "  # Paired benchmark for a research run: server and client each write a CSV that share\n"
        "  # the SAME <prefix> and line up for the identical 60 s scripted camera. On the GPU host:\n"
        "  #   rr-server 9000 1280 720 100000000 1 tcp --benchmark run1\n"
        "  # then on this client (drives the script, writes its CSV, then quits when it ends):\n"
        "  %s 127.0.0.1 9000 \"\" tcp --benchmark run1\n"
        "  # -> run1-<date>-rr-server-...csv (host) + run1-<date>-rr-client-...csv (here);\n"
        "  #    plot both with research/scripts/plot_benchmark.py. Add a large frames value\n"
        "  #    (e.g. 99999) before --benchmark to run this client headless.\n"
        "\n"
        "Reaching a server behind SSH (e.g. a Slurm job in a Pyxis/enroot container):\n"
        "  The server binds all interfaces (0.0.0.0) and enroot shares the host network, so it\n"
        "  listens on the compute node directly (no container port mapping needed). SSH forwards\n"
        "  TCP only, so tunnel in and connect with transport 'tcp' -- QUIC is UDP and will NOT\n"
        "  traverse an ssh -L tunnel. Find the node your job landed on with 'squeue -u $USER'\n"
        "  (the NODELIST column), or 'echo $SLURMD_NODENAME' from inside the job.\n"
        "\n"
        "  # From your laptop: forward local 9000 -> compute node, via the login node:\n"
        "  ssh -N -L 9000:<compute-node-name>:<port> <user>@<supercomputer-url>\n"
        "  # If the login node cannot reach compute-node ports, jump into the node instead:\n"
        "  ssh -N -J <user>@<supercomputer-url> <user>@<compute-node-name> -L 9000:localhost:<port>\n"
        "  # then, locally:\n"
        "  %s 127.0.0.1 9000 \"\" tcp\n"
        "\n"
        "  Concrete example (node gpu042, cluster hpc.example.edu, port 9000, token s3cret):\n"
        "    ssh -N -L 9000:gpu042:9000 alice@hpc.example.edu\n"
        "    %s 127.0.0.1 9000 s3cret tcp\n",
        prog, prog, prog, prog, prog, prog, prog, prog);
}

int main(int argc, char *argv[])
{
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // Split flags from positionals so --benchmark <csv> can appear anywhere.
    bool benchmark = false;
    const char *bench_csv = nullptr;
    std::vector<const char*> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        if (a == "--benchmark")
        {
            if (i + 1 >= argc) { fprintf(stderr, "missing csv file for --benchmark\n"); return EXIT_FAILURE; }
            benchmark = true;
            bench_csv = argv[++i];
            continue;
        }
        if (a == "--first-frame-timeout")
        {
            if (i + 1 >= argc) { fprintf(stderr, "missing SECONDS for --first-frame-timeout\n"); return EXIT_FAILURE; }
            g_first_frame_timeout_sec = std::atoi(argv[++i]);
            if (g_first_frame_timeout_sec <= 0)
            { fprintf(stderr, "invalid --first-frame-timeout (expected positive seconds)\n"); return EXIT_FAILURE; }
            continue;
        }
        if (a == "--window" || a == "--size")
        {
            if (i + 2 >= argc) { fprintf(stderr, "missing W H for %s (e.g. --window 1280 720)\n", a.c_str()); return EXIT_FAILURE; }
            g_win_w = std::atoi(argv[++i]);
            g_win_h = std::atoi(argv[++i]);
            if (g_win_w <= 0 || g_win_h <= 0)
            { fprintf(stderr, "invalid window size (expected positive W H, e.g. --window 1280 720)\n"); return EXIT_FAILURE; }
            continue;
        }
        pos.push_back(argv[i]);
    }

    std::string host  = (pos.size() >= 1) ? pos[0] : "127.0.0.1";
    std::string port  = (pos.size() >= 2) ? pos[1] : "9000";
    std::string token = (pos.size() >= 3) ? pos[2] : "";
    std::string mode  = (pos.size() >= 4) ? pos[3] : "auto"; // auto | quic | tcp
    int frames        = (pos.size() >= 5) ? std::atoi(pos[4]) : 0; // >0 => headless test mode
    g_dial_host = host; g_dial_port = port; // for the HUD server line

    // --benchmark carries a path+prefix; the CSV is opened with its auto-generated name once the
    // server's Hello identifies the server host and GPU (see bench_csv_open).
    if (bench_csv) { g_bench_prefix = bench_csv; g_hud.benchmark = true; }

    std::thread session(session_thread, host, port, token, mode);
    std::thread bench;
    if (benchmark) { bench = std::thread(bench_thread); }

    int rc = (frames > 0) ? run_headless(frames) : run_window();

    g.quit.store(true);
    if (bench.joinable())   { bench.join(); }
    if (session.joinable()) { session.join(); }
    if (g_csv) { fclose(g_csv); }
    return rc;
}
