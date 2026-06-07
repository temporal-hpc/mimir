// H.264 decoding remote client sample (step 3): verifies the encoded streaming path.
//
// Connects to run_remote_server (started with H.264 enabled), receives H.264 access units,
// decodes them with libavcodec, and — to stay verifiable without a display — saves a few frames
// to PPM and reports the received vs. decoded byte counts so the bandwidth saving is visible.
// It exercises the same control round-trip as run_remote_client (pause, then rotate). Depends
// only on the wire-protocol header + ffmpeg (no mimir/CUDA/Vulkan).
//
// Run from build/samples/:  ./run_remote_decode [host] [port]

#include <mimir/remote_protocol.hpp>
using namespace mimir::remote;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{

bool sendAll(int fd, const void *buf, size_t len)
{
    auto *p = static_cast<const char*>(buf);
    for (size_t sent = 0; sent < len; )
    {
        auto n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) { return false; }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool recvAll(int fd, void *buf, size_t len)
{
    auto *p = static_cast<char*>(buf);
    for (size_t got = 0; got < len; )
    {
        auto n = recv(fd, p + got, len - got, 0);
        if (n <= 0) { return false; }
        got += static_cast<size_t>(n);
    }
    return true;
}

void sendControl(int fd, ControlKind kind, float a = 0.f, float b = 0.f)
{
    ControlMsg msg{};
    msg.kind = static_cast<uint8_t>(kind);
    msg.a = a;
    msg.b = b;
    sendAll(fd, &msg, sizeof(msg));
}

// Sends the auth handshake (always first on the control channel; token may be empty).
bool sendAuth(int fd, const std::string& token)
{
    AuthMsg a{};
    a.magic = AUTH_MAGIC;
    size_t n = token.size() < TOKEN_MAX ? token.size() : static_cast<size_t>(TOKEN_MAX);
    std::memcpy(a.token, token.data(), n);
    return sendAll(fd, &a, sizeof(a));
}

// Saves a decoded BGRA buffer (already top-row-first) as a binary PPM.
void savePpm(const std::string& path, const unsigned char *bgra, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i)
    {
        f.put(static_cast<char>(bgra[i * 4 + 2])); // R
        f.put(static_cast<char>(bgra[i * 4 + 1])); // G
        f.put(static_cast<char>(bgra[i * 4 + 0])); // B
    }
    printf("  saved %s\n", path.c_str());
}

} // namespace

int main(int argc, char *argv[])
{
    std::string host  = (argc >= 2)? argv[1] : "127.0.0.1";
    std::string port  = (argc >= 3)? argv[2] : "9000";
    std::string token = (argc >= 4)? argv[3] : "";
    // Optional benchmark mode: receive N frames with the simulation left running (no pause/rotate
    // script, no PPM saves), so the server measures encode latency on real, changing content.
    int bench_frames  = (argc >= 5)? std::atoi(argv[4]) : 0;

    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0)
    {
        fprintf(stderr, "could not resolve %s:%s\n", host.c_str(), port.c_str());
        return EXIT_FAILURE;
    }
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0 || connect(fd, res->ai_addr, res->ai_addrlen) < 0)
    {
        fprintf(stderr, "could not connect to %s:%s\n", host.c_str(), port.c_str());
        return EXIT_FAILURE;
    }
    freeaddrinfo(res);

    // The server reads our AuthMsg before sending anything, so send it first.
    if (!sendAuth(fd, token)) { fprintf(stderr, "failed to send auth\n"); return EXIT_FAILURE; }

    Hello hello{};
    if (!recvAll(fd, &hello, sizeof(hello)) || hello.magic != PROTOCOL_MAGIC)
    {
        fprintf(stderr, "invalid server hello (rejected? wrong token?)\n");
        return EXIT_FAILURE;
    }
    int w = static_cast<int>(hello.width), h = static_cast<int>(hello.height);
    if (static_cast<Codec>(hello.codec) != Codec::H264)
    {
        fprintf(stderr, "server is not streaming H.264 (codec %u); start the server with "
            "H.264 enabled, or use run_remote_client for raw frames\n", hello.codec);
        close(fd);
        return EXIT_FAILURE;
    }
    printf("connected: stream is %dx%d (H.264)\n", w, h);

    // Set up the H.264 decoder + a BGRA conversion context.
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) { fprintf(stderr, "no H.264 decoder available\n"); close(fd); return EXIT_FAILURE; }
    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    if (avcodec_open2(ctx, codec, nullptr) < 0)
    {
        fprintf(stderr, "failed to open H.264 decoder\n");
        close(fd); return EXIT_FAILURE;
    }
    AVPacket *packet = av_packet_alloc();
    AVFrame  *frame  = av_frame_alloc();
    SwsContext *sws  = nullptr;
    std::vector<unsigned char> bgra(static_cast<size_t>(w) * h * 4);

    std::vector<unsigned char> au; // one H.264 access unit
    size_t total_encoded = 0;
    int received = 0, decoded = 0;
    bool done = false;
    while (!done)
    {
        FrameHeader header{};
        if (!recvAll(fd, &header, sizeof(header))) { break; }
        au.resize(header.size);
        if (!recvAll(fd, au.data(), header.size)) { break; }
        if (header.flags & FRAME_STATS)
        {
            Stats st{};
            if (au.size() >= sizeof(st)) { std::memcpy(&st, au.data(), sizeof(st)); }
            printf("[stats] %.1f fps, %u kbps, encode %.2f ms\n",
                st.fps_milli / 1000.0, st.kbps, st.encode_us / 1000.0);
            continue;
        }
        received++;
        total_encoded += header.size;

        // Scripted interaction, mirroring run_remote_client, to verify the control round-trip.
        // Skipped in benchmark mode so the simulation keeps advancing (changing content).
        if (bench_frames == 0)
        {
            if (received == 3)  { printf("pausing simulation\n"); sendControl(fd, ControlKind::TogglePause); }
            if (received == 7)  { printf("sending camera rotate\n"); sendControl(fd, ControlKind::CameraRotate, 150.f, 40.f); }
        }
        else if (received >= bench_frames)
        {
            sendControl(fd, ControlKind::Quit);
            done = true;
        }

        packet->data = au.data();
        packet->size = static_cast<int>(au.size());
        if (avcodec_send_packet(ctx, packet) < 0) { continue; }
        while (avcodec_receive_frame(ctx, frame) == 0)
        {
            if (!sws)
            {
                sws = sws_getContext(frame->width, frame->height,
                    static_cast<AVPixelFormat>(frame->format),
                    w, h, AV_PIX_FMT_BGRA, SWS_BILINEAR, nullptr, nullptr, nullptr);
            }
            uint8_t *dst[4]   = { bgra.data(), nullptr, nullptr, nullptr };
            int dst_stride[4] = { w * 4, 0, 0, 0 };
            sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst, dst_stride);
            decoded++;

            if (bench_frames == 0)
            {
                if (decoded == 1)  { savePpm("decode_frame0.ppm", bgra.data(), w, h); }
                if (decoded == 6)  { savePpm("decode_pre_rotate.ppm", bgra.data(), w, h); }
                if (decoded == 15)
                {
                    savePpm("decode_post_rotate.ppm", bgra.data(), w, h);
                    printf("sending quit\n");
                    sendControl(fd, ControlKind::Quit);
                    done = true;
                    break;
                }
            }
        }
    }

    const size_t raw_bytes = static_cast<size_t>(w) * h * 4 * static_cast<size_t>(received);
    printf("received %d access units (%zu encoded bytes), decoded %d frames\n",
        received, total_encoded, decoded);
    if (received > 0)
    {
        printf("avg %.1f KB/encoded-frame vs %.1f KB/raw-frame (%.1fx smaller)\n",
            total_encoded / 1024.0 / received,
            (raw_bytes / static_cast<double>(received)) / 1024.0,
            raw_bytes / static_cast<double>(total_encoded));
    }

    if (sws) { sws_freeContext(sws); }
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&ctx);
    close(fd);
    return (decoded > 0)? EXIT_SUCCESS : EXIT_FAILURE;
}
