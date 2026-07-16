// Raw-frame TCP streaming server for interactive remote rendering (step 2).
//
// Renders headless and streams each frame's raw pixels to a single connected client over TCP,
// while a background thread receives control events (camera, pause) and feeds them to the
// render loop. This is the bring-up transport: no encoding, no QUIC yet — it validates the
// frame ring, the readback path, the control round-trip, and the threading model before the
// NVENC/QUIC work in later steps. The transport is intentionally isolated here so it can be
// replaced behind the same surface.

#include "mimir/engine.hpp"
#include "mimir/framelimit.hpp" // getTargetFrameTime (optional --fps session cap)
#include "mimir/remote_protocol.hpp"
#include "mimir/transport.hpp"
#include "mimir/validation.hpp"

#include <algorithm>
#include <atomic> // decoupled sim thread: total_iter / stop / pause flags
#include <chrono>
#include <cmath> // std::sqrt for the encode-time std-dev
#include <cstdio> // benchmark stats CSV
#include <cstdlib>
#include <cstring>
#include <thread> // sovereign simulation thread (see serveRemote)
#include <vector>

#include <pwd.h>    // getpwuid: server identity announced in Hello
#include <unistd.h> // gethostname, geteuid

#ifdef MIMIR_HAVE_FFMPEG
#include <cuda.h>         // cuCtxGetCurrent: share mimir's CUDA context with ffmpeg
#include <cuda_runtime.h> // cudaMemcpy2D for the zero-copy NVENC path
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/cpu.h> // av_cpu_count: CPU threads for the software H.264 fallback
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#endif

namespace mimir
{

namespace
{

#ifdef MIMIR_HAVE_FFMPEG
// H.264 encoder with two input paths:
//   - zero-copy (preferred, h264_nvenc only): the rendered frame stays on the GPU. The caller
//     supplies a CUDA device pointer to the BGRA pixels (from mapFrameToCuda); we copy them
//     device->device into an AV_PIX_FMT_CUDA frame and NVENC does BGRA->NV12 + encode on-GPU.
//     No pixel data ever touches host memory.
//   - host fallback (libx264, or if CUDA setup fails): BGRA host bytes -> YUV420P via libswscale.
// The encoder picks h264_nvenc, falling back to libx264.
struct H264Encoder
{
    AVCodecContext *ctx = nullptr;
    SwsContext     *sws = nullptr;       // host path: BGRA->YUV420P conversion
    AVFrame        *frame = nullptr;     // host path: the YUV420P frame
    AVPacket       *packet = nullptr;
    AVBufferRef    *hw_device = nullptr; // zero-copy path
    AVBufferRef    *hw_frames = nullptr; // zero-copy path
    int width = 0, height = 0;
    int64_t pts = 0;
    bool zero_copy = false;
    int sw_threads = 0;  // libx264 CPU thread count (0 on the NVENC/GPU path)

    bool init(int w, int h, int fps, int bitrate_kbps)
    {
        width = w; height = h;
        // Prefer NVENC, unless MIMIR_FORCE_HOST_ENCODE pins the host path (before/after benchmarks).
        const bool force_host = std::getenv("MIMIR_FORCE_HOST_ENCODE") != nullptr;
        const AVCodec *nvenc = force_host ? nullptr : avcodec_find_encoder_by_name("h264_nvenc");
        if (nvenc && tryOpen(nvenc, "h264_nvenc", true, w, h, fps, bitrate_kbps)) { return true; }
        if (nvenc)
        {
            // NVENC is present in ffmpeg but could not open here — typically a datacenter GPU with
            // no encode ASIC (A100/H100). Retry on the CPU so the stream stays compressed instead
            // of falling all the way back to multi-MB raw frames.
            spdlog::warn("remote: NVENC unavailable on this GPU; falling back to software H.264");
            teardown();
        }
        // Software H.264: libx264 (preferred) or openh264, whichever the ffmpeg build carries.
        for (const char *sw : {"libx264", "libopenh264"})
        {
            const AVCodec *codec = avcodec_find_encoder_by_name(sw);
            if (codec && tryOpen(codec, sw, false, w, h, fps, bitrate_kbps)) { return true; }
            teardown();
        }
        spdlog::error("remote: no usable H.264 encoder (no NVENC ASIC and no software encoder)");
        return false;
    }

    // Attempts one specific encoder. On any failure it leaves partial state for the caller to
    // teardown() before the next attempt, and returns false.
    bool tryOpen(const AVCodec *codec, const char *name, bool is_nvenc,
        int w, int h, int fps, int bitrate_kbps)
    {
        // For NVENC, try to set up a CUDA hardware frames pool so we can feed on-GPU BGRA frames.
        if (is_nvenc) { zero_copy = initCudaFrames(w, h); }

        ctx = avcodec_alloc_context3(codec);
        if (!ctx) { return false; }
        ctx->width       = w;
        ctx->height      = h;
        ctx->time_base   = AVRational{1, fps};
        ctx->framerate   = AVRational{fps, 1};
        ctx->pix_fmt     = zero_copy ? AV_PIX_FMT_CUDA : AV_PIX_FMT_YUV420P;
        ctx->bit_rate    = static_cast<int64_t>(bitrate_kbps) * 1000;
        ctx->gop_size    = fps * 2;
        ctx->max_b_frames = 0;
        if (zero_copy) { ctx->hw_frames_ctx = av_buffer_ref(hw_frames); }
        // Low-latency options differ per encoder; set the ones each understands.
        sw_threads = 0; // 0 marks the GPU/NVENC path; set below for the software path
        if (is_nvenc)
        {
            av_opt_set(ctx->priv_data, "tune", "ll", 0);       // nvenc: low latency
            av_opt_set(ctx->priv_data, "preset", "p4", 0);     // nvenc preset (balanced)
            av_opt_set(ctx->priv_data, "delay", "0", 0);       // emit each frame immediately
            av_opt_set(ctx->priv_data, "forced-idr", "1", 0);  // forced I frames become real IDRs
        }
        else
        {
            av_opt_set(ctx->priv_data, "tune", "zerolatency", 0); // libx264: low latency
            // ultrafast (not fast): the CPU encode is the bottleneck on a no-NVENC GPU, and the
            // stream has huge headroom (~500x compression), so trade ratio for speed.
            av_opt_set(ctx->priv_data, "preset", "ultrafast", 0);
            // Use the cores available to this process. av_cpu_count() honors CPU affinity, so on a
            // cluster it reflects the SLURM/cgroup allocation, not the whole node. x264's sliced
            // threading stops helping (and warns) past 16 threads, so cap there. FF_THREAD_SLICE
            // parallelizes within a frame (no frame-buffering latency), matching zerolatency.
            sw_threads = std::min(av_cpu_count(), 16);
            ctx->thread_count = sw_threads;
            ctx->thread_type  = FF_THREAD_SLICE;
        }

        if (avcodec_open2(ctx, codec, nullptr) < 0) { return false; }
        packet = av_packet_alloc();
        if (!packet) { return false; }
        if (!zero_copy)
        {
            // Host path: YUV420P frame fed by libswscale.
            frame = av_frame_alloc();
            frame->format = ctx->pix_fmt;
            frame->width  = w;
            frame->height = h;
            av_frame_get_buffer(frame, 0);
            sws = sws_getContext(w, h, AV_PIX_FMT_BGRA, w, h, AV_PIX_FMT_YUV420P,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!sws || !frame) { return false; }
        }
        if (zero_copy)
        {
            spdlog::info("remote: H.264 encoder '{}' {}x{} @ {} kbps (zero-copy CUDA/NVENC)",
                name, w, h, bitrate_kbps);
        }
        else
        {
            spdlog::info("remote: H.264 encoder '{}' {}x{} @ {} kbps (host readback, {} CPU threads)",
                name, w, h, bitrate_kbps, sw_threads);
        }
        return true;
    }

    // Sets up a CUDA hwdevice + a BGRA frames pool, so NVENC can take on-GPU frames. Shares
    // mimir's already-current CUDA context (rather than letting ffmpeg create/retain the primary
    // context, which clashes with the flags mimir initialized it with). Returns false on any
    // failure, in which case the caller falls back to the host readback path.
    bool initCudaFrames(int w, int h)
    {
        CUcontext cur = nullptr;
        if (cuCtxGetCurrent(&cur) != CUDA_SUCCESS || cur == nullptr) { return false; }
        hw_device = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
        if (!hw_device) { return false; }
        auto *dctx = reinterpret_cast<AVHWDeviceContext*>(hw_device->data);
        auto *cudactx = reinterpret_cast<AVCUDADeviceContext*>(dctx->hwctx);
        cudactx->cuda_ctx = cur;
        if (av_hwdevice_ctx_init(hw_device) < 0)
        {
            av_buffer_unref(&hw_device);
            return false;
        }
        hw_frames = av_hwframe_ctx_alloc(hw_device);
        if (!hw_frames) { return false; }
        auto *fctx = reinterpret_cast<AVHWFramesContext*>(hw_frames->data);
        fctx->format    = AV_PIX_FMT_CUDA;
        fctx->sw_format = AV_PIX_FMT_BGRA;   // NVENC converts BGRA->NV12 internally on-GPU
        fctx->width     = w;
        fctx->height    = h;
        if (av_hwframe_ctx_init(hw_frames) < 0)
        {
            av_buffer_unref(&hw_frames);
            return false;
        }
        return true;
    }

    bool last_keyframe = false; // whether the most recent encode produced an IDR/keyframe AU
    // When set, the next encoded frame is forced to be an IDR (a decode restart point). Used
    // when a client on the unreliable datagram path lost a frame, or when the server itself
    // dropped one under congestion; cleared once consumed.
    bool force_idr = false;

    // Drains encoded packets into out. Returns false on a fatal encoder error.
    bool drain(std::vector<unsigned char>& out)
    {
        last_keyframe = false;
        for (;;)
        {
            int r = avcodec_receive_packet(ctx, packet);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) { break; }
            if (r < 0) { return false; }
            if (packet->flags & AV_PKT_FLAG_KEY) { last_keyframe = true; }
            out.insert(out.end(), packet->data, packet->data + packet->size);
            av_packet_unref(packet);
        }
        return true;
    }

    // Host path: encodes one BGRA host frame.
    bool encode(const unsigned char *bgra, std::vector<unsigned char>& out)
    {
        out.clear();
        if (av_frame_make_writable(frame) < 0) { return false; }
        const uint8_t *src[4] = { bgra, nullptr, nullptr, nullptr };
        int stride[4] = { width * 4, 0, 0, 0 };
        sws_scale(sws, src, stride, 0, height, frame->data, frame->linesize);
        frame->pict_type = force_idr ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
        force_idr = false;
        frame->pts = pts++;
        if (avcodec_send_frame(ctx, frame) < 0) { return false; }
        return drain(out);
    }

    // Zero-copy path: encodes one BGRA frame already in CUDA device memory (packed, stride=w*4).
    bool encodeCuda(const void *bgra_dev, std::vector<unsigned char>& out)
    {
        out.clear();
        AVFrame *hwf = av_frame_alloc();
        if (!hwf) { return false; }
        if (av_hwframe_get_buffer(hw_frames, hwf, 0) < 0) { av_frame_free(&hwf); return false; }
        // Device-to-device copy from the Vulkan-exported BGRA buffer into the NVENC input frame.
        cudaError_t cerr = cudaMemcpy2D(hwf->data[0], static_cast<size_t>(hwf->linesize[0]),
            bgra_dev, static_cast<size_t>(width) * 4,
            static_cast<size_t>(width) * 4, static_cast<size_t>(height),
            cudaMemcpyDeviceToDevice);
        if (cerr != cudaSuccess)
        {
            spdlog::error("remote: cudaMemcpy2D into NVENC frame failed: {}", cudaGetErrorString(cerr));
            av_frame_free(&hwf);
            return false;
        }
        if (force_idr) { hwf->pict_type = AV_PICTURE_TYPE_I; force_idr = false; }
        hwf->pts = pts++;
        int r = avcodec_send_frame(ctx, hwf);
        av_frame_free(&hwf);
        if (r < 0) { return false; }
        return drain(out);
    }

    // Frees all ffmpeg state and resets to a fresh (re-initializable) condition.
    void teardown()
    {
        if (sws)       { sws_freeContext(sws); sws = nullptr; }
        if (frame)     { av_frame_free(&frame); }       // nulls frame
        if (packet)    { av_packet_free(&packet); }     // nulls packet
        if (ctx)       { avcodec_free_context(&ctx); }  // nulls ctx
        if (hw_frames) { av_buffer_unref(&hw_frames); } // nulls hw_frames
        if (hw_device) { av_buffer_unref(&hw_device); } // nulls hw_device
        zero_copy = false;
        last_keyframe = false;
        force_idr = false;
        pts = 0;
    }

    // Rebuilds the encoder for a new resolution (used on resize). Returns false on failure.
    bool reinit(int w, int h, int fps, int bitrate_kbps)
    {
        teardown();
        return init(w, h, fps, bitrate_kbps);
    }

    ~H264Encoder() { teardown(); }
};
#endif // MIMIR_HAVE_FFMPEG

// Stamps the server's account and hostname into a Hello, so the client can show where the
// simulation is running. Best effort: fields stay empty ("") if the system won't say.
void fillServerIdentity(remote::Hello& hello)
{
    const char *user = std::getenv("USER");
    if (!user || !user[0]) { if (passwd *pw = getpwuid(geteuid())) { user = pw->pw_name; } }
    if (user) { std::strncpy(hello.user, user, sizeof(hello.user) - 1); }
    char host[remote::HOST_MAX]{};
    if (gethostname(host, sizeof(host) - 1) == 0) { std::memcpy(hello.host, host, sizeof(host)); }
    // GPU model doing the rendering/encoding, for the client HUD (best effort).
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        // %.*s bounds the copy to the field width (safe truncate + guaranteed NUL).
        std::snprintf(hello.gpu, sizeof(hello.gpu), "%.*s",
            static_cast<int>(sizeof(hello.gpu)) - 1, prop.name);
    }
}

// Human-scaled particles/second (G/M suffix) for the sim/stats lines. particles/s = the sim's
// particle count times its step rate; at billion-particle scale it reaches 10^11+, so a fixed
// suffix keeps it readable and is a better large-N throughput signal than steps/s (which falls
// as N grows even though the GPU is doing more work).
std::string formatParticleRate(double parts_per_sec)
{
    char buf[48];
    if (parts_per_sec >= 1e9)      { std::snprintf(buf, sizeof(buf), "%.2f Gpart/s", parts_per_sec / 1e9); }
    else if (parts_per_sec >= 1e6) { std::snprintf(buf, sizeof(buf), "%.1f Mpart/s", parts_per_sec / 1e6); }
    else                           { std::snprintf(buf, sizeof(buf), "%.0f part/s", parts_per_sec); }
    return buf;
}

// Human-scaled particle COUNT (K/M/G suffix, decimal) for the sim/stats lines and HUD, so a
// billion reads as "1.07 G" instead of "1073741824". Mirrors formatParticleRate's scaling.
std::string formatParticleCount(size_t n)
{
    char buf[32];
    const double d = static_cast<double>(n);
    if (n >= 1000000000ull) { std::snprintf(buf, sizeof(buf), "%.2f G", d / 1e9); }
    else if (n >= 1000000ull) { std::snprintf(buf, sizeof(buf), "%.1f M", d / 1e6); }
    else if (n >= 1000ull)    { std::snprintf(buf, sizeof(buf), "%.1f K", d / 1e3); }
    else                      { std::snprintf(buf, sizeof(buf), "%zu", n); }
    return buf;
}

// Live whole-device VRAM (used/total, GiB labeled GB to match the startup banner) via
// cudaMemGetInfo -- redundant ground truth that includes EVERYTHING on the GPU at that instant
// (CUDA particles + Vulkan render targets + the path-tracing BVH/instance buffers built during
// prepare()), unlike the pre-serve startup estimate which is taken before the RT scene exists.
std::string formatVram()
{
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0) { return "VRAM n/a"; }
    const double gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "VRAM %.1f/%.0f GB",
        static_cast<double>(total_b - free_b) * gib, static_cast<double>(total_b) * gib);
    return buf;
}

} // namespace

void MimirInstance::serveRemote(uint16_t port, std::function<void(void)> compute,
    size_t max_iters, bool use_h264, remote::TransportKind kind, std::string token,
    int bitrate_kbps, std::string stats_csv, int target_fps, int steps_per_frame)
{
    prepare();

    // Total particles the sim advances each step = the sum of the views' element counts.
    // Use element_count (the true 64-bit particle total in BOTH point and instanced-mesh modes),
    // NOT draw_count: draw_count is clamped to UINT32_MAX (and is the icosphere index count in mesh
    // mode), so it would report 4.29 G for any scene past 2^32. Captured once for the particles/s
    // throughput metric on the heartbeat/stats lines and the client HUD's scene-size readout.
    size_t particle_total = 0;
    for (auto *v : views) { if (v != nullptr) { particle_total += v->element_count; } }

    // One-shot honest footprint: prepare() has now built everything, including the path-tracing
    // BVH + instance buffers that the pre-serve startup estimate in rr-server is taken before and
    // therefore misses entirely. This is the real whole-device number; the heartbeat/stats lines
    // refresh it as the run proceeds.
    spdlog::info("remote: GPU memory after setup: {} ({} particles)", formatVram(), particle_total);

    // Frame/simulation coupling. Decoupled (steps_per_frame <= 0): the sim runs on its own
    // thread and frames sample the latest state — the viewer never slows the run. Lockstep
    // (steps_per_frame >= 1): this consumer advances exactly N steps then renders one frame,
    // sequentially, so frames are tear-free and deterministic (recording/reproducing) but the
    // fps cap and a slow client throttle the sim too. See the two branches below.
    const bool decoupled = (steps_per_frame <= 0);

    // The simulation runs on its own sovereign thread (spawned below); this loop is the
    // render+encode+send CONSUMER. So target_fps paces the *frame* cadence here, not the sim:
    // it is a pixels-on-the-wire rate (bandwidth budget + encoder rate-control framerate),
    // decoupled from how fast the sim advances. The limiter is frameStall() at the end of
    // renderFrame (engine.cpp), which now only this consumer thread hits — the sim never calls
    // renderFrame, so capping fps never throttles steps/s. target_fps == 0 means uncapped:
    // stream at the natural render+encode+send rate, bounded only by real backpressure (TCP) or
    // the datagram drop policy (QUIC).
    // Pace the frame cadence in this consumer loop (frameStall below), NOT inside renderFrame, so
    // the per-frame render time we measure/log excludes the fps-cap wait. 0 = uncapped.
    options.present.enable_fps_limit = false;
    const int64_t frame_period_ns = getTargetFrameTime(target_fps > 0, target_fps);
    // Rate-control framerate for the encoder: the cap when set, the conventional 60 otherwise.
    const int encoder_fps = target_fps > 0 ? target_fps : 60;

    // Camera model. Fly (options.camera_control == Fly, i.e. rr-server --fly): first-person
    // free-look + WASD, driven by the client's CameraLook/CameraMove events and rendered on the
    // engine's Fly (camera-to-world) branch. Otherwise the default trackball (orbit/zoom/pan).
    // The client is told which via the Hello flags so it adapts its input. prepare() left the
    // camera in the trackball default pose, so seed a sensible fly pose here: eye back on +z,
    // yaw 180 so forward is -z looking at the scene origin (the setFlyLook convention).
    const bool fly = (options.camera_control == CameraControl::Fly);
    if (fly)
    {
        camera.setPosition(glm::vec3(0.f, 0.f, 2.85f));
        camera.setRotation(glm::vec3(0.f, 180.f, 0.f));
        camera.setFlyLook();
    }

    // Step counter advanced by the sim thread, read by this consumer for the HUD/stats/logs.
    std::atomic<size_t> total_iter{0};
    bool stop = false;

    // Optional benchmark log: one row per telemetry window, mirroring the [stats] lines, with
    // time relative to server start. stats_csv is a path+prefix; the full name is assembled once
    // the first client connects (it carries the client hostname) as
    //   <prefix>-<date>-rr-server-c<client>-s<server>-<gpu>.csv
    // so the server's and client's CSVs for a run pair up by name. Sessions share the file.
    FILE *csv = nullptr;
    std::string server_host, server_gpu; // this server's identity, for the CSV name
    if (!stats_csv.empty())
    {
        char host[remote::HOST_MAX]{};
        if (gethostname(host, sizeof(host) - 1) == 0) { server_host = host; }
        int dev = 0; cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        {
            server_gpu = prop.name;
        }
    }
    const auto serve_start = std::chrono::steady_clock::now();

    // Bind the listener once; nullptr means a fatal bind error (port taken, no QUIC support).
    // From here on the simulation is sovereign: it advances continuously whether or not anyone
    // is watching, and clients attach/detach without ever pausing it — the long-running-job
    // monitoring model (connect from home, look around, disconnect, reconnect tomorrow).
    auto listener = (kind == remote::TransportKind::Quic)
        ? remote::makeQuicListener(port, token)
        : remote::makeTcpListener(port, token);
    if (!listener)
    {
        if (csv) { fclose(csv); }
        return;
    }
    spdlog::info("remote: simulation runs with or without a viewer; "
        "clients may connect and disconnect at any time");

    // ── Sovereign simulation thread (decoupled mode only) ───────────────────────────────────
    // The sim advances compute() as fast as the GPU allows, forever, independent of the frame
    // cadence and of whether anyone is watching — the monitor must not perturb the run (no
    // observer effect). This mirrors displayAsync's "unsynchronized" mode: the consumer thread
    // below renders whatever the latest buffer holds, accepting a torn-latest read as the price
    // of not slowing the science. TogglePause (from the viewer) pauses only this thread; the
    // consumer keeps streaming the frozen state (and the path tracer converges while paused).
    // In lockstep mode no thread is spawned: the consumer drives compute() inline instead.
    std::atomic<bool> sim_stop{false};
    std::atomic<bool> sim_paused{false};
    std::thread sim_thread;
    if (decoupled)
    {
        sim_thread = std::thread([&]()
        {
            while (!sim_stop.load(std::memory_order_acquire))
            {
                if (max_iters != 0 && total_iter.load(std::memory_order_relaxed) >= max_iters) { break; }
                if (sim_paused.load(std::memory_order_acquire))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    continue;
                }
                compute();
                total_iter.fetch_add(1, std::memory_order_release);
            }
        });
    }

    // Sim steps that had advanced when the previous viewer left, so the next client's session
    // can report how far the run moved while unwatched.
    size_t last_session_end_iter = 0;

    // Unwatched proof-of-life: a [sim] progress line every few seconds while nobody is
    // connected (step count + free-run rate). While a client is connected the step counter
    // rides the once-per-second [stats] line instead, so nothing floods.
    constexpr auto kSimLogPeriod = std::chrono::seconds(2);
    auto sim_log_time  = std::chrono::steady_clock::now();
    size_t sim_log_iter = total_iter.load();

    while (!stop)
    {
        if (max_iters != 0 && total_iter.load() >= max_iters) { break; }

        // No viewer. Decoupled: the sim thread is already advancing on its own, so here we only
        // watch for a dialing client and emit the heartbeat, sleeping briefly so the accept poll
        // doesn't busy-spin. Lockstep: there is no sim thread, so advance the sim inline at full
        // speed (render is skipped while unwatched, exactly like the classic free-run).
        std::unique_ptr<remote::Transport> transport = listener->poll();
        if (!transport)
        {
            if (!decoupled && !(max_iters != 0 && total_iter.load() >= max_iters))
            {
                compute();
                total_iter.fetch_add(1, std::memory_order_release);
            }
            const size_t iters = total_iter.load(std::memory_order_acquire);
            const auto sim_now = std::chrono::steady_clock::now();
            if (sim_now - sim_log_time >= kSimLogPeriod)
            {
                const double secs = std::chrono::duration<double>(sim_now - sim_log_time).count();
                const double rate = static_cast<double>(iters - sim_log_iter) / secs;
                const std::string prate = formatParticleRate(rate * static_cast<double>(particle_total));
                const std::string pcount = formatParticleCount(particle_total);
                const std::string vram = formatVram();
                if (max_iters != 0)
                {
                    spdlog::info("[sim] step {} of {} ({:.1f}%) | {} particles | {:.0f} steps/s | {} | {} | no viewer",
                        iters, max_iters,
                        100.0 * static_cast<double>(iters) / static_cast<double>(max_iters),
                        pcount, rate, prate, vram);
                }
                else
                {
                    spdlog::info("[sim] step {} | {} particles | {:.0f} steps/s | {} | {} | no viewer",
                        iters, pcount, rate, prate, vram);
                }
                sim_log_time = sim_now;
                sim_log_iter = iters;
            }
            // Decoupled mode busy-waits on the accept poll, so yield a moment; lockstep is
            // already paced by the inline compute() above.
            if (decoupled) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
            continue;
        }
        const size_t adv = total_iter.load() - last_session_end_iter;
        if (adv > 0)
        {
            spdlog::info("remote: simulation advanced {} steps while unwatched", adv);
        }

        // A client connected: build a fresh session. The encoder is rebuilt per session, so a
        // newly-joined client always starts on a clean IDR (keyframe-on-join).
        // Open the benchmark CSV now (first client only): its name needs the client's hostname.
        if (!stats_csv.empty() && !csv)
        {
            const std::string path = remote::benchmarkCsvPath(
                stats_csv, "server", transport->peerName(), server_host, server_gpu);
            csv = fopen(path.c_str(), "w");
            if (csv) { fprintf(csv, "time_s,frame,fps,steps_s,kbps,encode_ms\n"); fflush(csv);
                       spdlog::info("remote: benchmark CSV -> {}", path); }
            else { spdlog::warn("remote: cannot open stats csv '{}'", path); }
        }
        uint32_t width  = swapchain.extent.width;
        uint32_t height = swapchain.extent.height;

        // Decide the codec for this session. H.264 needs ffmpeg; otherwise stream raw.
        remote::Codec codec = remote::Codec::RawBGRA;
#ifdef MIMIR_HAVE_FFMPEG
        H264Encoder encoder;
        if (use_h264)
        {
            if (encoder.init(static_cast<int>(width), static_cast<int>(height), encoder_fps, bitrate_kbps))
            {
                codec = remote::Codec::H264;
            }
            else { spdlog::warn("remote: H.264 encoder unavailable, falling back to raw frames"); }
        }
#else
        if (use_h264) { spdlog::warn("remote: built without ffmpeg, falling back to raw frames"); }
#endif

        // Greet the client with the stream geometry.
        remote::Hello hello{
            .magic  = remote::PROTOCOL_MAGIC,
            .width  = width,
            .height = height,
            .format = static_cast<uint32_t>(remote::PixelFormat::BGRA8),
            .codec  = static_cast<uint32_t>(codec),
            .user   = {},
            .host   = {},
            .flags  = fly ? remote::HELLO_CAMERA_FLY : 0u, // tell the client to drive a Fly camera
            .gpu    = {},
        };
        fillServerIdentity(hello);
        if (!transport->sendVideo(&hello, sizeof(hello))) { continue; } // client vanished; re-listen

        // Unreliable video: when the client negotiated QUIC DATAGRAM support, H.264 frames go
        // out as never-retransmitted datagrams (a lost frame is skipped; the client requests an
        // IDR to resume). Raw frames stay on the reliable stream: at ~8 MB each, any single lost
        // fragment would discard the whole frame, so unreliability buys nothing there. Hello,
        // Stats and re-Hello always use the reliable stream.
        const bool dgram_video = (codec == remote::Codec::H264) && transport->unreliableVideoReady();
        if (dgram_video)
        {
            spdlog::info("remote: video over unreliable QUIC datagrams "
                "(lost frames are dropped, never retransmitted)");
        }

        // Newest client input/heartbeat timestamp, echoed once in the next frame so the client
        // can measure end-to-end latency on its own clock.
        uint32_t latest_stamp = 0;

        bool paused = false;
        bool client_gone = false;
        // Latest sim step reflected in a rendered frame. SIZE_MAX forces the first frame of the
        // session to reset the path-trace accumulator; thereafter the accumulator resets only
        // when the sim has actually advanced since our last frame (see the render call below).
        size_t last_render_iter = SIZE_MAX;
        std::vector<unsigned char> frame;
        std::vector<remote::ControlMsg> events;
#ifdef MIMIR_HAVE_FFMPEG
        std::vector<unsigned char> encoded;
#endif
        // Per-frame production latency (for benchmarking); first frames skipped as warmup.
        constexpr size_t kWarmup = 5;
        size_t produced_frames = 0, timed_count = 0;
        double enc_sum_ms = 0.0, enc_min_ms = 1e30, enc_max_ms = 0.0;

        // Telemetry window: a Stats message is sent to the client roughly once per second.
        auto win_start = std::chrono::steady_clock::now();
        size_t win_frames = 0, win_bytes = 0, session_frames = 0;
        size_t win_start_iter = total_iter.load(); // sim step count at the window's start
        double win_enc_us = 0.0, win_enc_us_sq = 0.0; // sum and sum-of-squares, for mean + std
        double win_render_ms = 0.0;                   // sum of per-frame GPU render time, for the mean
        // Path-tracing GPU-timestamp split, bucketed by the readback frame's actual build mode so the
        // times and their refit/rebuild/skip counts describe the same frames (see readTimings). The
        // build phase is broken into its three sub-phases: the AABB writer and the TLAS rebuild are
        // mode-independent (run identically on refit and rebuild frames), but the BLAS sub-phase is
        // the mode-sensitive part -- a full rebuild is ~2x a refit -- so it is kept per mode. Trace is
        // summed over every traced frame (skips still trace, with ~0 build).
        double win_aabb_ms = 0.0, win_tlas_ms = 0.0, win_trace_ms = 0.0;
        double win_refit_blas_ms = 0.0, win_rebuild_blas_ms = 0.0;
        uint64_t win_refit_n = 0, win_rebuild_n = 0, win_skip_n = 0;

        while (true)
        {
            if (max_iters != 0 && total_iter.load() >= max_iters) { stop = true; break; }

            // Drain control events; pollControl returns false once the client is gone.
            events.clear();
            if (!transport->pollControl(events)) { client_gone = true; break; }
            bool quit = false;
            bool want_resize = false;
            bool want_idr = false;
            int resize_w = 0, resize_h = 0;
            const auto speed = camera.rotation_speed;
            // Fly-camera input helpers (used only when `fly`), mirroring the on-screen Fly camera
            // (engine.cpp updateCamera / window.cpp): mouse-look about the eye and WASD movement
            // along the current view basis. setFlyLook rebuilds a roll-free camera-to-world view
            // from yaw/pitch at the eye -- the branch the path tracer decodes in Fly mode.
            const float sens  = options.mouse_sensitivity;                 // degrees per pixel
            const float mstep = options.camera_move_speed * (1.f / 60.f);  // units per move event
            auto flyLook = [&](float dx, float dy)
            {
                camera.rotation.y += dx * sens;   // drag right -> look right
                camera.rotation.x -= dy * sens;   // drag up    -> look up
                camera.rotation.x = std::clamp(camera.rotation.x, -89.9f, 89.9f);
                camera.setFlyLook();
            };
            auto flyMove = [&](float strafe, float forward)
            {
                const glm::vec3 fwd = glm::vec3(camera.matrices.view[2]); // world look dir (col 2)
                // cross(fwd, up), NOT matrices.view[0]: setLookAt's stored right is the opposite
                // (the trap the WASD handler documents), so this keeps strafe uninverted.
                const glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.f, 1.f, 0.f)));
                glm::vec3 dir = strafe * right + forward * fwd;
                if (glm::dot(dir, dir) > 0.f)
                {
                    camera.position += glm::normalize(dir) * mstep;
                    camera.setFlyLook();
                }
            };
            for (const auto& ev : events)
            {
                if (ev.stamp_ms != 0) { latest_stamp = ev.stamp_ms; } // newest wins (in order)
                switch (static_cast<remote::ControlKind>(ev.kind))
                {
                    case remote::ControlKind::CameraRotate:
                        // Fly: a mouse drag turns the gaze; trackball: it orbits the scene.
                        // Pitch is -ev.b so drag-up tilts the scene's top toward the viewer
                        // (matches drag direction); yaw stays -ev.a.
                        if (fly) { flyLook(ev.a, ev.b); }
                        else { camera.rotate(glm::vec3(-ev.b * speed, -ev.a * speed, 0.f)); }
                        break;
                    case remote::ControlKind::CameraLook:
                        // Fly: mouse-look about the eye (setFlyLook). Trackball: an in-place gaze
                        // turn about the eye (freeLook, world-to-view) -- same sign as the orbit.
                        if (fly) { flyLook(ev.a, ev.b); }
                        else { camera.freeLook(-ev.a * speed, ev.b * speed); }
                        break;
                    case remote::ControlKind::CameraMove:
                        // WASD flythrough (Fly only); a no-op for a trackball server.
                        if (fly) { flyMove(ev.a, ev.b); }
                        break;
                    case remote::ControlKind::CameraZoom:
                        if (!fly) { camera.translate(glm::vec3(0.f, 0.f, ev.a * 0.005f)); }
                        break;
                    case remote::ControlKind::CameraPan:
                        if (!fly) { camera.translate(glm::vec3(-ev.a * 0.01f, -ev.b * 0.01f, 0.f)); }
                        break;
                    case remote::ControlKind::TogglePause:
                        // Pause the sovereign sim thread; the consumer keeps streaming the
                        // frozen scene (and the path tracer converges on it).
                        paused = !paused;
                        sim_paused.store(paused, std::memory_order_release);
                        break;
                    case remote::ControlKind::Quit:
                        quit = true; break;
                    case remote::ControlKind::Resize:
                        want_resize = true;
                        resize_w = static_cast<int>(ev.a);
                        resize_h = static_cast<int>(ev.b);
                        break;
                    case remote::ControlKind::RequestKeyframe:
                        want_idr = true; break;
                    default: break;
                }
            }
            if (quit) { break; }
#ifdef MIMIR_HAVE_FFMPEG
            if (want_idr && codec == remote::Codec::H264) { encoder.force_idr = true; }
#else
            (void)want_idr; // raw frames are always self-contained
#endif

            // Apply a requested resolution change: rebuild the offscreen targets and encoder, then
            // re-announce the geometry to the client via a framed Hello (FRAME_HELLO). This is a
            // dormant capability: the bundled rr-client never sends Resize (a remote viewer
            // stretches the frame to its window instead of renegotiating resolution), but the
            // server-side path is wired and tested for clients that do want it.
            if (want_resize)
            {
                int nw = std::clamp(resize_w, 16, 7680) & ~1; // H.264 needs even dimensions
                int nh = std::clamp(resize_h, 16, 7680) & ~1;
                if (static_cast<uint32_t>(nw) != width || static_cast<uint32_t>(nh) != height)
                {
                    vkDeviceWaitIdle(device);
                    options.window.size = { nw, nh };
                    freeFrameCudaBuffer();
                    recreateGraphics();
                    width  = swapchain.extent.width;
                    height = swapchain.extent.height;
                    bool ok = true;
#ifdef MIMIR_HAVE_FFMPEG
                    if (codec == remote::Codec::H264)
                    {
                        ok = encoder.reinit(static_cast<int>(width), static_cast<int>(height), encoder_fps, bitrate_kbps);
                    }
#endif
                    remote::Hello rehello{
                        .magic  = remote::PROTOCOL_MAGIC,
                        .width  = width,
                        .height = height,
                        .format = static_cast<uint32_t>(remote::PixelFormat::BGRA8),
                        .codec  = static_cast<uint32_t>(codec),
                        .user   = {},
                        .host   = {},
                        .flags  = fly ? remote::HELLO_CAMERA_FLY : 0u,
                        .gpu    = {},
                    };
                    fillServerIdentity(rehello);
                    remote::FrameHeader hh{ .size = static_cast<uint32_t>(sizeof(rehello)),
                        .flags = remote::FRAME_HELLO, .echo_stamp = 0 };
                    if (!ok || !transport->sendVideo(&hh, sizeof(hh))
                            || !transport->sendVideo(&rehello, sizeof(rehello)))
                    {
                        client_gone = true; break;
                    }
                    spdlog::info("remote: resized stream to {}x{}", width, height);
                }
            }

            if (decoupled)
            {
                // The sim thread advances compute() independently; sample the latest state. Reset
                // path-trace accumulation only when the sim actually moved since our last frame,
                // so a paused (or slow) sim lets the accumulator converge on the static scene.
                const size_t it_now = total_iter.load(std::memory_order_acquire);
                if (it_now != last_render_iter) { pt_scene_dirty = true; last_render_iter = it_now; }
            }
            else if (!paused)
            {
                // Lockstep: advance exactly steps_per_frame steps, then render one frame — all
                // sequential on this thread, so the buffer is never read mid-write (tear-free).
                for (int s = 0; s < steps_per_frame; ++s)
                {
                    if (max_iters != 0 && total_iter.load() >= max_iters) { stop = true; break; }
                    compute();
                    total_iter.fetch_add(1, std::memory_order_relaxed);
                }
                pt_scene_dirty = true; // the sim moved, so reset the path-trace accumulator
            }
            if (stop) { break; } // hit max_iters mid-batch
            const auto render_t0 = std::chrono::steady_clock::now();
            renderFrame();
            vkDeviceWaitIdle(device); // ensure the frame is finished before readback

            const unsigned char *payload = nullptr;
            size_t payload_size = 0;
            bool produced = false;
            uint32_t flags = 0;
            const auto enc_t0 = std::chrono::steady_clock::now();
            // GPU render time for this frame (renderFrame + wait-for-idle), excluding the sim step.
            const double render_ms =
                std::chrono::duration<double, std::milli>(enc_t0 - render_t0).count();

#ifdef MIMIR_HAVE_FFMPEG
            // Zero-copy H.264: encode straight from the on-GPU frame, no host readback.
            if (codec == remote::Codec::H264 && encoder.zero_copy)
            {
                void *cuda_bgra = mapFrameToCuda();
                if (!cuda_bgra || !encoder.encodeCuda(cuda_bgra, encoded))
                {
                    spdlog::error("remote: H.264 zero-copy encode failed");
                    client_gone = true; break;
                }
                if (encoded.empty()) { continue; } // buffered; nothing to send this frame
                payload = encoded.data();
                payload_size = encoded.size();
                if (encoder.last_keyframe) { flags |= remote::FRAME_KEYFRAME; }
                produced = true;
            }
#endif
            // Host path: read the frame back, then either send raw or H.264-encode on the CPU.
            if (!produced)
            {
                readFrameBytes(frame);
                payload = frame.data();
                payload_size = frame.size();
                flags |= remote::FRAME_KEYFRAME; // raw frames are self-contained
#ifdef MIMIR_HAVE_FFMPEG
                if (codec == remote::Codec::H264)
                {
                    if (!encoder.encode(frame.data(), encoded))
                    {
                        spdlog::error("remote: H.264 encode failed");
                        client_gone = true; break;
                    }
                    if (encoded.empty()) { continue; }
                    payload = encoded.data();
                    payload_size = encoded.size();
                    flags = encoder.last_keyframe ? remote::FRAME_KEYFRAME : 0u;
                }
#endif
            }

            const auto enc_t1 = std::chrono::steady_clock::now();
            const double enc_ms = std::chrono::duration<double, std::milli>(enc_t1 - enc_t0).count();
            if (produced_frames++ >= kWarmup)
            {
                enc_sum_ms += enc_ms;
                enc_min_ms = std::min(enc_min_ms, enc_ms);
                enc_max_ms = std::max(enc_max_ms, enc_ms);
                ++timed_count;
            }

            // Echo the newest client stamp on exactly one frame (0 = nothing new to echo).
            const uint32_t echo = latest_stamp;
            latest_stamp = 0;

            bool sent_this = true;
            if (dgram_video)
            {
                if (!transport->sendVideoUnreliable(payload, payload_size, flags, echo))
                {
                    // Congestion backlog: the transport dropped the frame instead of queueing
                    // latency. The client sees a frame-id gap; make the next frame an IDR so it
                    // can resume decoding without waiting for its keyframe request round trip.
#ifdef MIMIR_HAVE_FFMPEG
                    encoder.force_idr = true;
#endif
                    sent_this = false;
                }
            }
            else
            {
                remote::FrameHeader header{ .size = static_cast<uint32_t>(payload_size),
                    .flags = flags, .echo_stamp = echo };
                if (!transport->sendVideo(&header, sizeof(header)) ||
                    !transport->sendVideo(payload, payload_size))
                {
                    client_gone = true; break;
                }
            }

            // Pace to the target fps here (not inside renderFrame), so the render_ms measured above
            // stays pure -- the frame is already sent, so this is inter-frame wait, not latency.
            // No-op when uncapped (frame_period_ns == 0).
            frameStall(frame_period_ns);

            // Telemetry: once per second, report fps / bitrate / mean encode time to the client
            // and print the same summary to the server terminal.
            if (sent_this)
            {
                ++win_frames; ++session_frames;
                win_bytes += payload_size;
                const double enc_us = enc_ms * 1000.0;
                win_enc_us += enc_us;
                win_enc_us_sq += enc_us * enc_us;
                win_render_ms += render_ms;
                // Path tracing exposes a GPU-timestamp split of the render: last_aabb_ms / last_blas_ms /
                // last_tlas_ms are the three build sub-phases, last_trace_ms the vkCmdTraceRays, and
                // last_build_mode the mode that produced them. readTimings paired them all from the same
                // (FRAMES-old) frame, so the BLAS time is bucketed by that frame's own mode; the AABB and
                // TLAS sub-phases are mode-independent. Skipped until have_timings, so the session's first
                // FRAMES frames (readback not written yet, values still defaults) do not pollute the mix.
                if (rt_enabled && raytracing.have_timings)
                {
                    switch (raytracing.last_build_mode)
                    {
                        case RayTracingContext::BlasBuild::Refit:
                            win_aabb_ms += raytracing.last_aabb_ms;
                            win_refit_blas_ms += raytracing.last_blas_ms; ++win_refit_n;
                            win_tlas_ms += raytracing.last_tlas_ms;
                            break;
                        case RayTracingContext::BlasBuild::Rebuild:
                            win_aabb_ms += raytracing.last_aabb_ms;
                            win_rebuild_blas_ms += raytracing.last_blas_ms; ++win_rebuild_n;
                            win_tlas_ms += raytracing.last_tlas_ms;
                            break;
                        case RayTracingContext::BlasBuild::Skip:
                            ++win_skip_n; break; // skip build time is ~0 (adjacent timestamps)
                    }
                    win_trace_ms += raytracing.last_trace_ms;
                }
            }
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - win_start).count();
            if (elapsed >= 1.0 && win_frames > 0)
            {
                const size_t iters = total_iter.load(std::memory_order_acquire);
                const double enc_mean = win_enc_us / static_cast<double>(win_frames);
                const double enc_var  = win_enc_us_sq / static_cast<double>(win_frames)
                    - enc_mean * enc_mean;
                const double render_mean = win_render_ms / static_cast<double>(win_frames);
                remote::Stats st{
                    .frames    = static_cast<uint32_t>(session_frames),
                    .fps_milli = static_cast<uint32_t>(static_cast<double>(win_frames) / elapsed * 1000.0),
                    .kbps      = static_cast<uint32_t>(static_cast<double>(win_bytes) * 8.0 / 1000.0 / elapsed),
                    .encode_us = static_cast<uint32_t>(enc_mean),
                    .step       = iters,
                    .step_limit = max_iters,
                    .encode_std_us = static_cast<uint32_t>(std::sqrt(std::max(0.0, enc_var))),
                    .particle_count = 0,     // filled below, once sps is known
                    .particles_per_sec = 0,
                };
                // Per-frame sizes: what the render produced vs. what actually went on the wire.
                // With H.264 the ratio is the compression achieved; with raw frames it is 1.0x.
                const double raw_frame_kb  = static_cast<double>(width) * height * 4.0 / 1000.0;
                const double sent_frame_kb = static_cast<double>(win_bytes)
                    / static_cast<double>(win_frames) / 1000.0;
                // The sim step count rides this line while a client is connected (the [sim]
                // heartbeat covers the unwatched periods). steps/s is the sim's own rate over
                // this window — in decoupled mode independent of fps, in lockstep ~= fps * N.
                const double sps = static_cast<double>(iters - win_start_iter) / elapsed;
                st.particle_count = particle_total;
                st.particles_per_sec = static_cast<uint64_t>(sps * static_cast<double>(particle_total));
                char step_str[64];
                if (max_iters != 0)
                {
                    snprintf(step_str, sizeof(step_str), "%zu of %zu", iters, max_iters);
                }
                else
                {
                    snprintf(step_str, sizeof(step_str), "%zu", iters);
                }
                // Label the per-frame production cost: NVENC on the GPU, libx264 on N CPU threads,
                // or raw framebuffer readback. (encode_us spans readback+convert+encode; see below.)
                std::string prod_label = "readback";
#ifdef MIMIR_HAVE_FFMPEG
                if (codec == remote::Codec::H264)
                {
                    prod_label = encoder.zero_copy
                        ? std::string("encode (GPU)")
                        : "encode (" + std::to_string(encoder.sw_threads) + " CPU threads)";
                }
#endif
                const std::string prate = formatParticleRate(sps * static_cast<double>(particle_total));
                const std::string pcount = formatParticleCount(particle_total);
                const std::string vram = formatVram();
                // For path tracing, break the render time into its GPU-timestamped AS-build vs trace
                // phases so it is clear where the frame goes (at large N the build dominates).
                std::string rt_split;
                if (rt_enabled && !raytracing.have_timings)
                {
                    // Session warmup: the readback slots are not written for the first FRAMES frames.
                    rt_split = " (build timings warming up)";
                }
                else if (rt_enabled)
                {
                    // Per-window mix of the AS-build modes: cheap in-place refits vs full rebuilds vs
                    // skipped (unchanged-scene) frames. A healthy live stream is mostly refits with the
                    // occasional rebuild; a paused view is mostly skips. The build phase is split into
                    // aabb (writer) + blas + tlas so it is clear where the time goes; the BLAS part is
                    // shown per mode when a window mixes them (a full rebuild is ~2x a refit, so a blended
                    // mean would hide that). Skips have ~0 build and appear only in the count.
                    auto fmt1 = [](double v) {
                        char b[32]; snprintf(b, sizeof(b), "%.1f", v); return std::string(b);
                    };
                    const uint64_t build_n = win_refit_n + win_rebuild_n; // frames that built (not skips)
                    std::string build_part;
                    if (build_n)
                    {
                        // BLAS sub-phase: label per mode only when both occurred; the count triple
                        // disambiguates a single-mode window.
                        std::string blas;
                        if (win_refit_n && win_rebuild_n)
                        {
                            blas = "refit " + fmt1(win_refit_blas_ms / static_cast<double>(win_refit_n))
                                 + " / rebuild " + fmt1(win_rebuild_blas_ms / static_cast<double>(win_rebuild_n));
                        }
                        else if (win_refit_n)   { blas = fmt1(win_refit_blas_ms / static_cast<double>(win_refit_n)); }
                        else                    { blas = fmt1(win_rebuild_blas_ms / static_cast<double>(win_rebuild_n)); }
                        build_part = "aabb " + fmt1(win_aabb_ms / static_cast<double>(build_n))
                                   + " + blas " + blas
                                   + " + tlas " + fmt1(win_tlas_ms / static_cast<double>(build_n));
                    }
                    else { build_part = "no build"; } // window was all skips
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                        " (%s + trace %.1f ms | %llu refit, %llu rebuild, %llu skip)",
                        build_part.c_str(),
                        win_trace_ms / static_cast<double>(win_frames),
                        static_cast<unsigned long long>(win_refit_n),
                        static_cast<unsigned long long>(win_rebuild_n),
                        static_cast<unsigned long long>(win_skip_n));
                    rt_split = buf;
                }
                spdlog::info("[stats] step {} ({} particles, {:.0f} steps/s, {}) | frame {:6d} | {:5.1f} fps | "
                    "{:6d} kbps | {:5.2f} ms render{} | {:5.2f} ms {} | {:.0f} kB -> {:.0f} kB/frame ({:.1f}x smaller) | {}",
                    step_str,
                    pcount, sps, prate,
                    st.frames,
                    static_cast<double>(st.fps_milli) / 1000.0,
                    st.kbps,
                    render_mean,
                    rt_split,
                    static_cast<double>(st.encode_us) / 1000.0,
                    prod_label,
                    raw_frame_kb, sent_frame_kb,
                    sent_frame_kb > 0.0 ? raw_frame_kb / sent_frame_kb : 0.0,
                    vram);
                if (csv)
                {
                    fprintf(csv, "%.3f,%u,%.1f,%.1f,%u,%.3f\n",
                        std::chrono::duration<double>(now - serve_start).count(),
                        st.frames, static_cast<double>(st.fps_milli) / 1000.0, sps, st.kbps,
                        static_cast<double>(st.encode_us) / 1000.0);
                    fflush(csv);
                }
                remote::FrameHeader sh{ .size = static_cast<uint32_t>(sizeof(st)),
                    .flags = remote::FRAME_STATS, .echo_stamp = 0 };
                if (!transport->sendVideo(&sh, sizeof(sh)) || !transport->sendVideo(&st, sizeof(st)))
                {
                    client_gone = true; break;
                }
                win_start = now; win_frames = 0; win_bytes = 0;
                win_enc_us = 0.0; win_enc_us_sq = 0.0; win_render_ms = 0.0;
                win_aabb_ms = 0.0; win_tlas_ms = 0.0; win_trace_ms = 0.0;
                win_refit_blas_ms = 0.0; win_rebuild_blas_ms = 0.0;
                win_refit_n = 0; win_rebuild_n = 0; win_skip_n = 0;
                win_start_iter = iters;
            }
        }

        if (timed_count > 0)
        {
#ifdef MIMIR_HAVE_FFMPEG
            const char *path = (codec == remote::Codec::H264)
                ? (encoder.zero_copy ? "H.264 zero-copy CUDA/NVENC" : "H.264 host readback+libswscale")
                : "raw readback";
#else
            const char *path = "raw readback";
#endif
            spdlog::info("remote: frame production latency [{}] over {} frames: "
                "mean {:.2f} ms, min {:.2f} ms, max {:.2f} ms",
                path, timed_count, enc_sum_ms / static_cast<double>(timed_count),
                enc_min_ms, enc_max_ms);
        }
        spdlog::info("remote: client session ended after {} frames ({}) — simulation continues unwatched",
            session_frames, client_gone ? "disconnected" : "client quit");
        // A viewer just left: mark the step count so the next session can report how far the
        // (still-running) sim advances while unwatched. Restart the heartbeat window too so the
        // first unwatched [sim] rate isn't averaged over the streaming period.
        last_session_end_iter = total_iter.load();
        sim_log_time = std::chrono::steady_clock::now();
        sim_log_iter = last_session_end_iter;
        // Loop back: keep polling for the next client while the sim thread free-runs.
    }

    // Wind down the sovereign sim thread before returning (reached only when max_iters is set
    // and hit; an unbounded server loops forever above).
    sim_stop.store(true, std::memory_order_release);
    if (sim_thread.joinable()) { sim_thread.join(); }
    if (csv) { fclose(csv); }
}

} // namespace mimir
