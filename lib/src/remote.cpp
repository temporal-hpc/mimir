// Raw-frame TCP streaming server for interactive remote rendering (step 2).
//
// Renders headless and streams each frame's raw pixels to a single connected client over TCP,
// while a background thread receives control events (camera, pause) and feeds them to the
// render loop. This is the bring-up transport: no encoding, no QUIC yet — it validates the
// frame ring, the readback path, the control round-trip, and the threading model before the
// NVENC/QUIC work in later steps. The transport is intentionally isolated here so it can be
// replaced behind the same surface.

#include "mimir/engine.hpp"
#include "mimir/remote_protocol.hpp"
#include "mimir/validation.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#ifdef MIMIR_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#endif

namespace mimir
{

namespace
{

#ifdef MIMIR_HAVE_FFMPEG
// Minimal H.264 encoder: BGRA -> YUV420P (libswscale) -> H.264 access units (libavcodec).
// Prefers the hardware h264_nvenc encoder, falling back to software libx264.
struct H264Encoder
{
    AVCodecContext *ctx = nullptr;
    SwsContext     *sws = nullptr;
    AVFrame        *frame = nullptr;
    AVPacket       *packet = nullptr;
    int width = 0, height = 0;
    int64_t pts = 0;

    bool init(int w, int h, int fps, int bitrate_kbps)
    {
        width = w; height = h;
        const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
        const char *name = "h264_nvenc";
        if (!codec) { codec = avcodec_find_encoder_by_name("libx264"); name = "libx264"; }
        if (!codec) { codec = avcodec_find_encoder(AV_CODEC_ID_H264); name = "h264"; }
        if (!codec) { spdlog::error("remote: no H.264 encoder available"); return false; }

        ctx = avcodec_alloc_context3(codec);
        ctx->width       = w;
        ctx->height      = h;
        ctx->time_base   = AVRational{1, fps};
        ctx->framerate   = AVRational{fps, 1};
        ctx->pix_fmt     = AV_PIX_FMT_YUV420P;
        ctx->bit_rate    = static_cast<int64_t>(bitrate_kbps) * 1000;
        ctx->gop_size    = fps * 2;
        ctx->max_b_frames = 0;
        // Low-latency options differ per encoder; set the ones each understands (the others
        // would just log a warning). Interactive streaming wants no frame buffering.
        const bool is_nvenc = std::strcmp(name, "h264_nvenc") == 0;
        if (is_nvenc)
        {
            av_opt_set(ctx->priv_data, "tune", "ll", 0);       // nvenc: low latency
            av_opt_set(ctx->priv_data, "preset", "p4", 0);     // nvenc preset (balanced)
            av_opt_set(ctx->priv_data, "delay", "0", 0);       // emit each frame immediately
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
        frame = av_frame_alloc();
        frame->format = ctx->pix_fmt;
        frame->width  = w;
        frame->height = h;
        av_frame_get_buffer(frame, 0);
        packet = av_packet_alloc();
        sws = sws_getContext(w, h, AV_PIX_FMT_BGRA, w, h, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        bool ok = sws && frame && packet;
        if (ok) { spdlog::info("remote: H.264 encoder '{}' {}x{} @ {} kbps", name, w, h, bitrate_kbps); }
        return ok;
    }

    // Encodes one BGRA frame; appends the resulting H.264 access unit bytes to out.
    bool encode(const unsigned char *bgra, std::vector<unsigned char>& out)
    {
        out.clear();
        if (av_frame_make_writable(frame) < 0) { return false; }
        const uint8_t *src[4] = { bgra, nullptr, nullptr, nullptr };
        int stride[4] = { width * 4, 0, 0, 0 };
        sws_scale(sws, src, stride, 0, height, frame->data, frame->linesize);
        frame->pts = pts++;
        if (avcodec_send_frame(ctx, frame) < 0) { return false; }
        for (;;)
        {
            int r = avcodec_receive_packet(ctx, packet);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) { break; }
            if (r < 0) { return false; }
            out.insert(out.end(), packet->data, packet->data + packet->size);
            av_packet_unref(packet);
        }
        return true;
    }

    ~H264Encoder()
    {
        if (sws)    { sws_freeContext(sws); }
        if (frame)  { av_frame_free(&frame); }
        if (packet) { av_packet_free(&packet); }
        if (ctx)    { avcodec_free_context(&ctx); }
    }
};
#endif // MIMIR_HAVE_FFMPEG

// Sends exactly len bytes, looping over partial writes. Returns false on error/disconnect.
bool sendAll(int fd, const void *buf, size_t len)
{
    auto *p = static_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < len)
    {
        auto n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) { return false; }
        sent += static_cast<size_t>(n);
    }
    return true;
}

// Receives exactly len bytes. Returns false on error/disconnect.
bool recvAll(int fd, void *buf, size_t len)
{
    auto *p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < len)
    {
        auto n = recv(fd, p + got, len - got, 0);
        if (n <= 0) { return false; }
        got += static_cast<size_t>(n);
    }
    return true;
}

} // namespace

void MimirInstance::serveRemote(uint16_t port, std::function<void(void)> compute,
    size_t max_iters, bool use_h264)
{
    prepare();

    const uint32_t width  = swapchain.extent.width;
    const uint32_t height = swapchain.extent.height;

    // Decide the codec. H.264 is only available when built with ffmpeg; otherwise fall back
    // to raw frames so the stream still works (the client is told which codec via Hello).
    remote::Codec codec = remote::Codec::RawBGRA;
#ifdef MIMIR_HAVE_FFMPEG
    H264Encoder encoder;
    if (use_h264)
    {
        if (encoder.init(static_cast<int>(width), static_cast<int>(height), 60, 8000))
        {
            codec = remote::Codec::H264;
        }
        else
        {
            spdlog::warn("remote: H.264 encoder unavailable, falling back to raw frames");
        }
    }
#else
    if (use_h264)
    {
        spdlog::warn("remote: built without ffmpeg, falling back to raw frames");
    }
#endif

    // Open a listening socket and wait for a single client to connect.
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { spdlog::error("remote: failed to create socket"); return; }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(port);
    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        spdlog::error("remote: failed to bind port {}", port);
        close(listen_fd);
        return;
    }
    listen(listen_fd, 1);
    spdlog::info("remote: waiting for a client on port {}", port);

    int client = accept(listen_fd, nullptr, nullptr);
    if (client < 0) { spdlog::error("remote: accept failed"); close(listen_fd); return; }
    int one = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); // low latency
    spdlog::info("remote: client connected");

    // Greet the client with the stream geometry.
    remote::Hello hello{
        .magic  = remote::PROTOCOL_MAGIC,
        .width  = width,
        .height = height,
        .format = static_cast<uint32_t>(remote::PixelFormat::BGRA8),
        .codec  = static_cast<uint32_t>(codec),
    };
    if (!sendAll(client, &hello, sizeof(hello)))
    {
        close(client); close(listen_fd); return;
    }

    // Control events are received on a background thread and applied on the render thread,
    // so the camera is only ever touched by one thread.
    std::atomic<bool> connected{true};
    std::mutex queue_mutex;
    std::deque<remote::ControlMsg> events;
    std::thread receiver([&]
    {
        remote::ControlMsg msg{};
        while (connected.load())
        {
            if (!recvAll(client, &msg, sizeof(msg))) { connected.store(false); break; }
            if (static_cast<remote::ControlKind>(msg.kind) == remote::ControlKind::Quit)
            {
                connected.store(false); break;
            }
            std::lock_guard<std::mutex> lock(queue_mutex);
            events.push_back(msg);
        }
    });

    bool paused = false;
    size_t iter = 0;
    std::vector<unsigned char> frame;
#ifdef MIMIR_HAVE_FFMPEG
    std::vector<unsigned char> encoded;
#endif
    while (connected.load() && (max_iters == 0 || iter < max_iters))
    {
        // Drain pending control events and apply them to camera / pause state.
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            while (!events.empty())
            {
                auto ev = events.front();
                events.pop_front();
                auto speed = camera.rotation_speed;
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
                    default: break;
                }
            }
        }

        if (!paused) { compute(); iter++; }
        renderFrame();
        vkDeviceWaitIdle(device); // ensure the frame is finished before readback

        readFrameBytes(frame);

        const unsigned char *payload = frame.data();
        size_t payload_size = frame.size();
#ifdef MIMIR_HAVE_FFMPEG
        if (codec == remote::Codec::H264)
        {
            if (!encoder.encode(frame.data(), encoded))
            {
                spdlog::error("remote: H.264 encode failed");
                connected.store(false);
                break;
            }
            // An access unit may be empty while the encoder buffers; skip sending in that case.
            if (encoded.empty()) { continue; }
            payload = encoded.data();
            payload_size = encoded.size();
        }
#endif
        remote::FrameHeader header{ .size = static_cast<uint32_t>(payload_size) };
        if (!sendAll(client, &header, sizeof(header)) ||
            !sendAll(client, payload, payload_size))
        {
            connected.store(false);
            break;
        }
    }

    connected.store(false);
    shutdown(client, SHUT_RDWR);
    close(client);
    close(listen_fd);
    if (receiver.joinable()) { receiver.join(); }
    spdlog::info("remote: client session ended after {} frames", iter);
}

} // namespace mimir
