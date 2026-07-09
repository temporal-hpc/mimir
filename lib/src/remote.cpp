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
#include <chrono>
#include <cstdio> // benchmark stats CSV
#include <cstdlib>
#include <cstring>
#include <vector>

#include <pwd.h>    // getpwuid: server identity announced in Hello
#include <unistd.h> // gethostname, geteuid

#ifdef MIMIR_HAVE_FFMPEG
#include <cuda.h>         // cuCtxGetCurrent: share mimir's CUDA context with ffmpeg
#include <cuda_runtime.h> // cudaMemcpy2D for the zero-copy NVENC path
extern "C" {
#include <libavcodec/avcodec.h>
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
    SwsContext     *sws = nullptr;       // host path only
    AVFrame        *frame = nullptr;     // host path: the YUV420P frame
    AVPacket       *packet = nullptr;
    AVBufferRef    *hw_device = nullptr; // zero-copy path
    AVBufferRef    *hw_frames = nullptr; // zero-copy path
    int width = 0, height = 0;
    int64_t pts = 0;
    bool zero_copy = false;

    bool init(int w, int h, int fps, int bitrate_kbps)
    {
        width = w; height = h;
        const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
        const char *name = "h264_nvenc";
        bool is_nvenc = true;
        if (!codec) { codec = avcodec_find_encoder_by_name("libx264"); name = "libx264"; is_nvenc = false; }
        if (!codec) { codec = avcodec_find_encoder(AV_CODEC_ID_H264); name = "h264"; is_nvenc = false; }
        if (!codec) { spdlog::error("remote: no H.264 encoder available"); return false; }

        // For NVENC, try to set up a CUDA hardware frames pool so we can feed on-GPU BGRA frames.
        // MIMIR_FORCE_HOST_ENCODE forces the host readback path (for before/after benchmarking).
        if (is_nvenc && std::getenv("MIMIR_FORCE_HOST_ENCODE") == nullptr)
        {
            zero_copy = initCudaFrames(w, h);
        }

        ctx = avcodec_alloc_context3(codec);
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
            av_opt_set(ctx->priv_data, "preset", "fast", 0);
        }

        if (avcodec_open2(ctx, codec, nullptr) < 0)
        {
            spdlog::error("remote: failed to open H.264 encoder");
            return false;
        }
        packet = av_packet_alloc();
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
        spdlog::info("remote: H.264 encoder '{}' {}x{} @ {} kbps ({})",
            name, w, h, bitrate_kbps, zero_copy ? "zero-copy CUDA/NVENC" : "host readback");
        return packet != nullptr;
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
}

} // namespace

void MimirInstance::serveRemote(uint16_t port, std::function<void(void)> compute,
    size_t max_iters, bool use_h264, remote::TransportKind kind, std::string token,
    int bitrate_kbps, std::string stats_csv, int target_fps)
{
    prepare();

    // By default no minimum frame time is imposed on the streaming loop. The fps limiter is a
    // windowed-present option (defaults to 60), but here there is no display to pace: sessions
    // run at the natural render+encode+send rate, throttled only by real backpressure (TCP) or
    // shed by the datagram drop policy (QUIC). An explicit target_fps > 0 re-enables the cap —
    // useful to hold a fixed bandwidth budget, since the encoder's rate control budgets bits
    // per frame at its configured framerate.
    if (target_fps > 0)
    {
        options.present.enable_fps_limit  = true;
        options.present.target_fps        = target_fps;
        options.present.target_frame_time = getTargetFrameTime(true, target_fps);
    }
    else
    {
        options.present.enable_fps_limit = false;
    }
    // Rate-control framerate for the encoder: the cap when set, the conventional 60 otherwise.
    const int encoder_fps = target_fps > 0 ? target_fps : 60;

    size_t total_iter = 0;
    bool stop = false;

    // Optional benchmark log: one row per telemetry window, mirroring the [stats] lines, with
    // time relative to server start (sessions from the reconnect loop share the file).
    FILE *csv = nullptr;
    if (!stats_csv.empty())
    {
        csv = fopen(stats_csv.c_str(), "w");
        if (csv) { fprintf(csv, "time_s,frame,fps,kbps,encode_ms\n"); }
        else { spdlog::warn("remote: cannot open stats csv '{}'", stats_csv); }
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

    size_t unwatched_iters = 0; // sim steps advanced since the last viewer left

    // Unwatched proof-of-life: a [sim] progress line every few seconds while nobody is
    // connected (step count + free-run rate). While a client is connected the step counter
    // rides the once-per-second [stats] line instead, so nothing floods.
    constexpr auto kSimLogPeriod = std::chrono::seconds(2);
    auto sim_log_time  = std::chrono::steady_clock::now();
    size_t sim_log_iter = total_iter;

    while (!stop)
    {
        if (max_iters != 0 && total_iter >= max_iters) { break; }

        // No viewer: check for a dialing client, then advance the simulation at full speed —
        // skipping render/encode entirely (the fps limiter lives in renderFrame, so unwatched
        // compute is not throttled either).
        std::unique_ptr<remote::Transport> transport = listener->poll();
        if (!transport)
        {
            compute(); ++total_iter; pt_scene_dirty = true; ++unwatched_iters;
            const auto sim_now = std::chrono::steady_clock::now();
            if (sim_now - sim_log_time >= kSimLogPeriod)
            {
                const double secs = std::chrono::duration<double>(sim_now - sim_log_time).count();
                const double rate = static_cast<double>(total_iter - sim_log_iter) / secs;
                if (max_iters != 0)
                {
                    spdlog::info("[sim] step {} of {} ({:.1f}%) | {:.0f} steps/s | no viewer",
                        total_iter, max_iters,
                        100.0 * static_cast<double>(total_iter) / static_cast<double>(max_iters),
                        rate);
                }
                else
                {
                    spdlog::info("[sim] step {} | {:.0f} steps/s | no viewer", total_iter, rate);
                }
                sim_log_time = sim_now;
                sim_log_iter = total_iter;
            }
            continue;
        }
        if (unwatched_iters > 0)
        {
            spdlog::info("remote: simulation advanced {} steps while unwatched", unwatched_iters);
            unwatched_iters = 0;
        }

        // A client connected: build a fresh session. The encoder is rebuilt per session, so a
        // newly-joined client always starts on a clean IDR (keyframe-on-join).
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
        double win_enc_us = 0.0;

        while (true)
        {
            if (max_iters != 0 && total_iter >= max_iters) { stop = true; break; }

            // Drain control events; pollControl returns false once the client is gone.
            events.clear();
            if (!transport->pollControl(events)) { client_gone = true; break; }
            bool quit = false;
            bool want_resize = false;
            bool want_idr = false;
            int resize_w = 0, resize_h = 0;
            const auto speed = camera.rotation_speed;
            for (const auto& ev : events)
            {
                if (ev.stamp_ms != 0) { latest_stamp = ev.stamp_ms; } // newest wins (in order)
                switch (static_cast<remote::ControlKind>(ev.kind))
                {
                    case remote::ControlKind::CameraRotate:
                        camera.rotate(glm::vec3(ev.b * speed, -ev.a * speed, 0.f)); break;
                    case remote::ControlKind::CameraZoom:
                        camera.translate(glm::vec3(0.f, 0.f, ev.a * 0.005f)); break;
                    case remote::ControlKind::CameraPan:
                        camera.translate(glm::vec3(-ev.a * 0.01f, -ev.b * 0.01f, 0.f)); break;
                    case remote::ControlKind::TogglePause:
                        paused = !paused; break;
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

            if (!paused) { compute(); ++total_iter; pt_scene_dirty = true; }
            renderFrame();
            vkDeviceWaitIdle(device); // ensure the frame is finished before readback

            const unsigned char *payload = nullptr;
            size_t payload_size = 0;
            bool produced = false;
            uint32_t flags = 0;
            const auto enc_t0 = std::chrono::steady_clock::now();

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

            // Telemetry: once per second, report fps / bitrate / mean encode time to the client
            // and print the same summary to the server terminal.
            if (sent_this)
            {
                ++win_frames; ++session_frames;
                win_bytes += payload_size;
                win_enc_us += enc_ms * 1000.0;
            }
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - win_start).count();
            if (elapsed >= 1.0 && win_frames > 0)
            {
                remote::Stats st{
                    .frames    = static_cast<uint32_t>(session_frames),
                    .fps_milli = static_cast<uint32_t>(static_cast<double>(win_frames) / elapsed * 1000.0),
                    .kbps      = static_cast<uint32_t>(static_cast<double>(win_bytes) * 8.0 / 1000.0 / elapsed),
                    .encode_us = static_cast<uint32_t>(win_enc_us / static_cast<double>(win_frames)),
                    .step       = total_iter,
                    .step_limit = max_iters,
                };
                // Per-frame sizes: what the render produced vs. what actually went on the wire.
                // With H.264 the ratio is the compression achieved; with raw frames it is 1.0x.
                const double raw_frame_kb  = static_cast<double>(width) * height * 4.0 / 1000.0;
                const double sent_frame_kb = static_cast<double>(win_bytes)
                    / static_cast<double>(win_frames) / 1000.0;
                // The sim step count rides this line while a client is connected (the [sim]
                // heartbeat covers the unwatched periods).
                char step_str[64];
                if (max_iters != 0)
                {
                    snprintf(step_str, sizeof(step_str), "%zu of %zu", total_iter, max_iters);
                }
                else
                {
                    snprintf(step_str, sizeof(step_str), "%zu", total_iter);
                }
                spdlog::info("[stats] step {} | frame {:6d} | {:5.1f} fps | {:6d} kbps | "
                    "{:5.2f} ms {} | {:.0f} kB -> {:.0f} kB/frame ({:.1f}x smaller)",
                    step_str,
                    st.frames,
                    static_cast<double>(st.fps_milli) / 1000.0,
                    st.kbps,
                    static_cast<double>(st.encode_us) / 1000.0,
                    codec == remote::Codec::H264 ? "encode" : "readback",
                    raw_frame_kb, sent_frame_kb,
                    sent_frame_kb > 0.0 ? raw_frame_kb / sent_frame_kb : 0.0);
                if (csv)
                {
                    fprintf(csv, "%.3f,%u,%.1f,%u,%.3f\n",
                        std::chrono::duration<double>(now - serve_start).count(),
                        st.frames, static_cast<double>(st.fps_milli) / 1000.0, st.kbps,
                        static_cast<double>(st.encode_us) / 1000.0);
                    fflush(csv);
                }
                remote::FrameHeader sh{ .size = static_cast<uint32_t>(sizeof(st)),
                    .flags = remote::FRAME_STATS, .echo_stamp = 0 };
                if (!transport->sendVideo(&sh, sizeof(sh)) || !transport->sendVideo(&st, sizeof(st)))
                {
                    client_gone = true; break;
                }
                win_start = now; win_frames = 0; win_bytes = 0; win_enc_us = 0.0;
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
        // Restart the heartbeat window so the first unwatched [sim] rate isn't averaged over
        // the session's (throttled) streaming period.
        sim_log_time = std::chrono::steady_clock::now();
        sim_log_iter = total_iter;
        // Loop back: free-run the simulation and keep polling for the next client.
    }
    if (csv) { fclose(csv); }
}

} // namespace mimir
