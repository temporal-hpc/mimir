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
// Depends only on the wire-protocol header + ngtcp2 + OpenSSL + ffmpeg + GLFW/OpenGL (no mimir,
// CUDA, or Vulkan): it models a laptop-class thin client.
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
using namespace mimir::remote;

#include <ngtcp2/ngtcp2.h>
#include <ngtcp2/ngtcp2_crypto.h>
#include <ngtcp2/ngtcp2_crypto_ossl.h>
// ngtcp2_conn_get_expiry2 was added after 1.22.x; alias it to the original on older releases.
#ifndef HAVE_NGTCP2_CONN_GET_EXPIRY2
#  define ngtcp2_conn_get_expiry2 ngtcp2_conn_get_expiry
#endif

#include <openssl/ssl.h>
#include <openssl/rand.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include <GLFW/glfw3.h>
#include <GL/gl.h>

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

const unsigned char ALPN[] = {5, 'm', 'i', 'm', 'i', 'r'};

uint64_t now_ns()
{
    timespec tp{};
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return static_cast<uint64_t>(tp.tv_sec) * NGTCP2_SECONDS + static_cast<uint64_t>(tp.tv_nsec);
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

// Optional CSV time-series log (one row per server Stats window), for offline plotting.
FILE *g_csv = nullptr;

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
            dec_ms_sum += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
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
        dec_ms_sum += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
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
        if (len >= sizeof(st)) { std::memcpy(&st, payload, sizeof(st)); }
        // fps/kbps/encode come from the server (encode_us is ITS per-frame production time);
        // decode, end-to-end latency, loss and the size expansion are measured here, over the
        // frames since the last Stats.
        const double n       = dec.dec_frames > 0 ? static_cast<double>(dec.dec_frames) : 1.0;
        const double dec_ms  = dec.dec_ms_sum / n;
        const double recv_kb = static_cast<double>(dec.recv_bytes) / n / 1000.0;
        const double out_kb  = static_cast<double>(dec.out_bytes) / n / 1000.0;
        // Latency percentiles over this window's samples (heartbeats give ~20 samples/s).
        double lat_mean = 0.0, lat_p50 = 0.0, lat_p95 = 0.0, lat_max = 0.0;
        if (!dec.lat_ms.empty())
        {
            std::sort(dec.lat_ms.begin(), dec.lat_ms.end());
            for (double v : dec.lat_ms) { lat_mean += v; }
            lat_mean /= static_cast<double>(dec.lat_ms.size());
            lat_p50 = dec.lat_ms[dec.lat_ms.size() / 2];
            lat_p95 = dec.lat_ms[(dec.lat_ms.size() * 95) / 100];
            lat_max = dec.lat_ms.back();
        }
        const uint32_t ctrl = g.ctrl_sent.exchange(0);
        printf("[stats] %.1f fps, %u kbps | server %s %.2f ms | decode %.2f ms | "
            "latency %.1f ms (p95 %.1f) | %zu lost | %.0f kB -> %.0f kB/frame (%.1fx larger)\n",
            st.fps_milli / 1000.0, st.kbps,
            dec.stream_codec == Codec::H264 ? "encode" : "readback",
            st.encode_us / 1000.0, dec_ms, lat_mean, lat_p95, dec.lost,
            recv_kb, out_kb, recv_kb > 0.0 ? out_kb / recv_kb : 0.0);
        if (g_csv)
        {
            const char *phase = g_phase.load();
            if (phase[0] == '\0') { phase = ctrl > 0 ? "move" : "idle"; }
            fprintf(g_csv, "%.3f,%.1f,%u,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%zu,%u,%s\n",
                now_ms() / 1000.0, st.fps_milli / 1000.0, st.kbps,
                st.encode_us / 1000.0, dec_ms,
                lat_mean, lat_p50, lat_p95, lat_max, dec.lost, ctrl, phase);
            fflush(g_csv);
        }
        dec.dec_ms_sum = 0.0; dec.dec_frames = 0; dec.recv_bytes = 0; dec.out_bytes = 0;
        dec.lat_ms.clear(); dec.lost = 0;
        return;
    }
    if (flags & FRAME_HELLO)
    {
        // Dormant on the server by default, but handled for completeness: geometry changed.
        if (len >= sizeof(Hello)) { Hello h{}; std::memcpy(&h, payload, sizeof(h)); dec.set_geometry(h); }
        return;
    }
    dec.feed(payload, len);
    // End-to-end latency: client stamp -> (server queue+render+encode) -> network -> decoded
    // frame, all on this client's clock (the stamp merely round-tripped).
    if (echo_stamp != 0)
    {
        dec.lat_ms.push_back(static_cast<double>(now_ms() - echo_stamp));
    }
}

void fill_auth(AuthMsg& a, const std::string& token)
{
    a.magic = AUTH_MAGIC;
    size_t n = token.size() < TOKEN_MAX ? token.size() : static_cast<size_t>(TOKEN_MAX);
    std::memcpy(a.token, token.data(), n);
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
        fprintf(stderr, "TCP: invalid server hello (rejected? wrong token?)\n");
        close(fd); return false;
    }
    dec.set_geometry(hello);
    printf("connected over TCP: %ux%u (%s)\n", hello.width, hello.height,
        static_cast<Codec>(hello.codec) == Codec::H264 ? "H.264" : "raw");

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

int quic_get_new_cid(ngtcp2_conn*, ngtcp2_cid *cid, ngtcp2_stateless_reset_token *token,
    size_t cidlen, void*)
{
    if (RAND_bytes(cid->data, static_cast<int>(cidlen)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    cid->datalen = cidlen;
    if (RAND_bytes(token->data, sizeof(token->data)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    return 0;
}

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
        printf("connected over QUIC: %ux%u (%s)\n", hello.width, hello.height,
            static_cast<Codec>(hello.codec) == Codec::H264 ? "H.264" : "raw");
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
    cb.get_path_challenge_data2 = ngtcp2_crypto_get_path_challenge_data2_cb;
    cb.version_negotiation      = ngtcp2_crypto_version_negotiation_cb;
    cb.rand                     = quic_rand;
    cb.get_new_connection_id2   = quic_get_new_cid;
    cb.handshake_completed      = quic_handshake_done;
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

// ============================================================================================
// Session thread: pick transport, run until shutdown.
// ============================================================================================
void session_thread(std::string host, std::string port, std::string token, std::string mode)
{
    Decoder dec;
    dec.init();
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
    dec.destroy();
    g.running.store(false);
}

// ============================================================================================
// Window (interactive) and headless (test) front-ends.
// ============================================================================================
struct Input { double last_x = 0, last_y = 0; bool left = false, right = false, middle = false; };
Input g_input;

void cursor_cb(GLFWwindow*, double x, double y)
{
    float dx = static_cast<float>(g_input.last_x - x);
    float dy = static_cast<float>(g_input.last_y - y);
    g_input.last_x = x; g_input.last_y = y;
    if (g_input.left)   { ui_control(ControlKind::CameraRotate, dx, dy); }
    if (g_input.right)  { ui_control(ControlKind::CameraZoom, dy); }
    if (g_input.middle) { ui_control(ControlKind::CameraPan, dx, dy); }
}
void button_cb(GLFWwindow*, int button, int action, int)
{
    bool pressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)   { g_input.left = pressed; }
    if (button == GLFW_MOUSE_BUTTON_RIGHT)  { g_input.right = pressed; }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) { g_input.middle = pressed; }
}
void key_cb(GLFWwindow *win, int key, int, int action, int)
{
    if (action != GLFW_PRESS) { return; }
    if (key == GLFW_KEY_P) { ui_control(ControlKind::TogglePause); }
    if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(win, GLFW_TRUE); }
}

// Waits (up to ~10 s) for the session to deliver the first frame so we know the stream geometry.
bool wait_for_geometry(int& w, int& h)
{
    for (int i = 0; i < 1000 && g.running.load(); ++i)
    {
        { std::lock_guard<std::mutex> lock(g.frame_mtx); if (g.have_geometry) { w = g.w; h = g.h; return true; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

int run_window()
{
    int w = 0, h = 0;
    if (!wait_for_geometry(w, h)) { fprintf(stderr, "no stream received\n"); return EXIT_FAILURE; }

    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return EXIT_FAILURE; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // window is freely resizable; the frame is stretched
    GLFWwindow *window = glfwCreateWindow(w, h, "mimir rr-client", nullptr, nullptr);
    if (!window) { fprintf(stderr, "window creation failed\n"); glfwTerminate(); return EXIT_FAILURE; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetCursorPosCallback(window, cursor_cb);
    glfwSetMouseButtonCallback(window, button_cb);
    glfwSetKeyCallback(window, key_cb);

    std::vector<unsigned char> display;
    uint64_t shown_seq = 0;
    int cur_w = w, cur_h = h;
    while (!glfwWindowShouldClose(window) && g.running.load())
    {
        glfwPollEvents();
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
// Emits the same control stream every run at a steady 60 Hz "mouse" cadence: idle to establish
// a baseline, one full orbit around the cloud, dive inside it, look around, back out, idle
// again, then quit. Magnitudes are tuned to the engine's default camera (LookAt at z=-2.85 over
// a roughly unit-sized cloud): the zoom legs travel ~2.4 world units in and back out. Phases
// are published to g_phase so every CSV row is labeled for plot shading.
void bench_thread()
{
    // Wait for the stream to come up so the script timeline starts at the first frame.
    for (int i = 0; i < 1000 && g.running.load() && !g.quit.load(); ++i)
    {
        { std::lock_guard<std::mutex> lock(g.frame_mtx); if (g.have_geometry) { break; } }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    struct Step { const char *phase; double secs; ControlKind kind; float a, b; };
    static const Step script[] = {
        { "idle",     3.0, ControlKind::None,         0.0f, 0.f }, // baseline, no interaction
        { "orbit",    6.0, ControlKind::CameraRotate, 1.0f, 0.f }, // ~360 deg yaw around the cloud
        { "zoom_in",  3.0, ControlKind::CameraZoom,   2.7f, 0.f }, // outside -> inside the cloud
        { "inside",   3.0, ControlKind::CameraRotate, 1.0f, 0.f }, // look around from within
        { "zoom_out", 3.0, ControlKind::CameraZoom,  -2.7f, 0.f }, // back out to the start radius
        { "idle",     2.0, ControlKind::None,         0.0f, 0.f }, // settle, end-of-run baseline
    };

    printf("benchmark: scripted camera starting (20 s: idle/orbit/zoom_in/inside/zoom_out/idle)\n");
    for (const auto& s : script)
    {
        if (!g.running.load() || g.quit.load()) { break; }
        g_phase.store(s.phase);
        printf("benchmark: phase %-8s (%.0f s)\n", s.phase, s.secs);
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
        "Usage: %s [host] [port] [token] [transport] [frames] [--benchmark out.csv]\n"
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
        "Flags (order-independent):\n"
        "  --benchmark F  Drive the camera with a deterministic 20 s script (3 s idle, 6 s\n"
        "                 orbit, 3 s zoom into the cloud, 3 s look around inside, 3 s zoom\n"
        "                 out, 2 s idle), quit, and write the per-second telemetry time series\n"
        "                 to CSV file F (columns: time_s,fps,kbps,server_ms,decode_ms,\n"
        "                 lat_mean_ms,lat_p50_ms,lat_p95_ms,lat_max_ms,lost,ctrl_events,phase;\n"
        "                 'phase' labels the script phases for plot shading). The control\n"
        "                 stream is identical every run, so results from different servers are\n"
        "                 directly comparable. Pair with the server's --benchmark file.\n"
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
        "  %s 127.0.0.1 9000 \"\" tcp 10\n",
        prog, prog, prog, prog, prog);
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
        pos.push_back(argv[i]);
    }

    std::string host  = (pos.size() >= 1) ? pos[0] : "127.0.0.1";
    std::string port  = (pos.size() >= 2) ? pos[1] : "9000";
    std::string token = (pos.size() >= 3) ? pos[2] : "";
    std::string mode  = (pos.size() >= 4) ? pos[3] : "auto"; // auto | quic | tcp
    int frames        = (pos.size() >= 5) ? std::atoi(pos[4]) : 0; // >0 => headless test mode

    if (bench_csv)
    {
        g_csv = fopen(bench_csv, "w");
        if (!g_csv) { fprintf(stderr, "cannot open csv log '%s'\n", bench_csv); return EXIT_FAILURE; }
        fprintf(g_csv, "time_s,fps,kbps,server_ms,decode_ms,lat_mean_ms,lat_p50_ms,lat_p95_ms,"
            "lat_max_ms,lost,ctrl_events,phase\n");
    }

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
