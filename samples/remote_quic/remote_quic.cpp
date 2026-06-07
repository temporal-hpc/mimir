// Auto-fallback remote rendering client sample (step 4/5): a thin native client that prefers
// QUIC and falls back to TCP.
//
// Connects to run_remote_server, sends the auth handshake, receives the video stream (Hello +
// framed messages: frames or Stats telemetry), decodes H.264 (or accepts raw), and sends control
// events back. In auto mode it tries QUIC first (UDP+TLS+congestion control) and, if QUIC never
// comes up (UDP blocked, handshake timeout — e.g. an ssh -L tunnel), transparently falls back to
// TCP. To stay verifiable without a display it scripts an interaction (pause, then rotate), saves
// a few PPMs, and prints stats. Depends only on the wire-protocol header + ngtcp2 + OpenSSL +
// ffmpeg (no mimir/CUDA/Vulkan).
//
// Run from build/samples/:  ./run_remote_quic [host] [port] [token] [auto|quic|tcp]

#include <mimir/remote_protocol.hpp>
using namespace mimir::remote;

#include <ngtcp2/ngtcp2.h>
#include <ngtcp2/ngtcp2_crypto.h>
#include <ngtcp2/ngtcp2_crypto_ossl.h>

#include <openssl/ssl.h>
#include <openssl/rand.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <string>
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

void savePpm(const std::string& path, const unsigned char *bgra, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int i = 0; i < w * h; ++i)
    {
        f.put(static_cast<char>(bgra[i * 4 + 2]));
        f.put(static_cast<char>(bgra[i * 4 + 1]));
        f.put(static_cast<char>(bgra[i * 4 + 0]));
    }
    printf("  saved %s\n", path.c_str());
}

struct QuicClient
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
    bool quit = false;
    int64_t control_stream = -1;

    // When >= 0 the session is running over the TCP fallback instead of QUIC; control is sent
    // directly on this socket and video is read from it.
    int tcp_fd = -1;

    // Video assembly.
    std::vector<uint8_t> vbuf;
    bool got_hello = false;
    Hello hello{};

    // Pending control bytes to write on the control stream.
    std::vector<uint8_t> ctrl_out;
    size_t ctrl_off = 0;

    // H.264 decode state.
    const AVCodec *avcodec = nullptr;
    AVCodecContext *avctx = nullptr;
    AVPacket *avpkt = nullptr;
    AVFrame *avframe = nullptr;
    SwsContext *sws = nullptr;
    std::vector<unsigned char> bgra;

    int received = 0, decoded = 0;
    size_t total_encoded = 0;
};

QuicClient g_client;

ngtcp2_conn* get_conn_cb(ngtcp2_crypto_conn_ref *ref)
{
    return static_cast<QuicClient*>(ref->user_data)->conn;
}

void rand_cb(uint8_t *dest, size_t destlen, const ngtcp2_rand_ctx*)
{
    RAND_bytes(dest, static_cast<int>(destlen));
}

int get_new_cid_cb(ngtcp2_conn*, ngtcp2_cid *cid, ngtcp2_stateless_reset_token *token,
    size_t cidlen, void*)
{
    if (RAND_bytes(cid->data, static_cast<int>(cidlen)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    cid->datalen = cidlen;
    if (RAND_bytes(token->data, sizeof(token->data)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
    return 0;
}

int handshake_completed_cb(ngtcp2_conn*, void *user_data)
{
    static_cast<QuicClient*>(user_data)->handshake_done = true;
    return 0;
}

int recv_stream_data_cb(ngtcp2_conn *conn, uint32_t, int64_t stream_id, uint64_t,
    const uint8_t *data, size_t datalen, void *user_data, void*)
{
    auto *c = static_cast<QuicClient*>(user_data);
    c->vbuf.insert(c->vbuf.end(), data, data + datalen);
    ngtcp2_conn_extend_max_stream_offset(conn, stream_id, datalen);
    ngtcp2_conn_extend_max_offset(conn, datalen);
    return 0;
}

// Blocking send/recv of exactly len bytes (used by the TCP fallback path).
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
    for (size_t g = 0; g < len; )
    {
        ssize_t k = recv(fd, p + g, len - g, 0);
        if (k <= 0) { return false; }
        g += static_cast<size_t>(k);
    }
    return true;
}

// Sends a control event over whichever transport the session is using.
void send_ctrl(QuicClient *c, ControlKind kind, float a = 0.f, float b = 0.f)
{
    ControlMsg msg{};
    msg.kind = static_cast<uint8_t>(kind);
    msg.a = a;
    msg.b = b;
    if (c->tcp_fd >= 0)
    {
        tcpSendAll(c->tcp_fd, &msg, sizeof(msg));
        return;
    }
    auto *p = reinterpret_cast<const uint8_t*>(&msg);
    c->ctrl_out.insert(c->ctrl_out.end(), p, p + sizeof(msg));
}

// Queues the auth handshake as the first bytes on the control stream (token may be empty).
void queue_auth(QuicClient *c, const std::string& token)
{
    AuthMsg a{};
    a.magic = AUTH_MAGIC;
    size_t n = token.size() < TOKEN_MAX ? token.size() : static_cast<size_t>(TOKEN_MAX);
    std::memcpy(a.token, token.data(), n);
    auto *p = reinterpret_cast<const uint8_t*>(&a);
    c->ctrl_out.insert(c->ctrl_out.end(), p, p + sizeof(a));
}

// Decodes one H.264 access unit (or copies a raw frame) into c->bgra, running the verification
// script (save PPMs, send control, quit) keyed on the decoded-frame counter.
void on_frame(QuicClient *c, const uint8_t *payload, size_t len)
{
    c->received++;
    c->total_encoded += len;

    if (c->received == 3)  { printf("pausing simulation\n"); send_ctrl(c, ControlKind::TogglePause); }
    if (c->received == 7)  { printf("sending camera rotate\n"); send_ctrl(c, ControlKind::CameraRotate, 150.f, 40.f); }

    int w = static_cast<int>(c->hello.width), h = static_cast<int>(c->hello.height);

    auto handle_decoded = [&](const unsigned char *bgra)
    {
        c->decoded++;
        if (c->decoded == 1)  { savePpm("auto_frame0.ppm", bgra, w, h); }
        if (c->decoded == 6)  { savePpm("auto_pre_rotate.ppm", bgra, w, h); }
        if (c->decoded == 15)
        {
            savePpm("auto_post_rotate.ppm", bgra, w, h);
            printf("sending quit\n");
            send_ctrl(c, ControlKind::Quit);
            c->quit = true;
        }
    };

    if (static_cast<Codec>(c->hello.codec) == Codec::RawBGRA)
    {
        handle_decoded(payload);
        return;
    }

    // H.264: feed the access unit and pull decoded frames.
    c->avpkt->data = const_cast<uint8_t*>(payload);
    c->avpkt->size = static_cast<int>(len);
    if (avcodec_send_packet(c->avctx, c->avpkt) < 0) { return; }
    while (avcodec_receive_frame(c->avctx, c->avframe) == 0)
    {
        if (!c->sws)
        {
            c->sws = sws_getContext(c->avframe->width, c->avframe->height,
                static_cast<AVPixelFormat>(c->avframe->format), w, h, AV_PIX_FMT_BGRA,
                SWS_BILINEAR, nullptr, nullptr, nullptr);
        }
        uint8_t *dst[4]   = { c->bgra.data(), nullptr, nullptr, nullptr };
        int dst_stride[4] = { w * 4, 0, 0, 0 };
        sws_scale(c->sws, c->avframe->data, c->avframe->linesize, 0, c->avframe->height,
            dst, dst_stride);
        handle_decoded(c->bgra.data());
        if (c->quit) { break; }
    }
}

// Handles one video-channel message (stats telemetry or a frame), shared by both transports.
void handle_message(QuicClient *c, uint32_t flags, const uint8_t *payload, size_t len)
{
    if (flags & FRAME_STATS)
    {
        Stats st{};
        if (len >= sizeof(st)) { std::memcpy(&st, payload, sizeof(st)); }
        printf("[stats] %.1f fps, %u kbps, encode %.2f ms\n",
            st.fps_milli / 1000.0, st.kbps, st.encode_us / 1000.0);
        return;
    }
    on_frame(c, payload, len);
}

// Parses the QUIC video byte stream into the Hello header and length-prefixed messages.
void process_video(QuicClient *c)
{
    size_t pos = 0;
    if (!c->got_hello)
    {
        if (c->vbuf.size() < sizeof(Hello)) { return; }
        std::memcpy(&c->hello, c->vbuf.data(), sizeof(Hello));
        if (c->hello.magic != PROTOCOL_MAGIC)
        {
            fprintf(stderr, "invalid server hello over QUIC\n");
            c->quit = true;
            return;
        }
        c->got_hello = true;
        pos = sizeof(Hello);
        int w = static_cast<int>(c->hello.width), h = static_cast<int>(c->hello.height);
        c->bgra.assign(static_cast<size_t>(w) * h * 4, 0);
        printf("connected over QUIC: stream is %dx%d (%s)\n", w, h,
            static_cast<Codec>(c->hello.codec) == Codec::H264 ? "H.264" : "raw");
    }

    for (;;)
    {
        if (c->vbuf.size() - pos < sizeof(FrameHeader)) { break; }
        FrameHeader fh{};
        std::memcpy(&fh, c->vbuf.data() + pos, sizeof(FrameHeader));
        if (c->vbuf.size() - pos - sizeof(FrameHeader) < fh.size) { break; }
        handle_message(c, fh.flags, c->vbuf.data() + pos + sizeof(FrameHeader), fh.size);
        pos += sizeof(FrameHeader) + fh.size;
        if (c->quit) { break; }
    }
    // Drop the consumed prefix.
    if (pos > 0) { c->vbuf.erase(c->vbuf.begin(), c->vbuf.begin() + static_cast<long>(pos)); }
}

// TCP fallback session: connect, authenticate, then receive Hello + framed messages, reusing the
// same decoder and scripted control as the QUIC path. Returns true if it ran a session.
bool run_tcp(QuicClient *c, const char *host, const char *port, const std::string& token)
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
    c->tcp_fd = fd;

    AuthMsg auth{};
    auth.magic = AUTH_MAGIC;
    size_t tn = token.size() < TOKEN_MAX ? token.size() : static_cast<size_t>(TOKEN_MAX);
    std::memcpy(auth.token, token.data(), tn);
    if (!tcpSendAll(fd, &auth, sizeof(auth))) { close(fd); c->tcp_fd = -1; return false; }

    if (!tcpRecvAll(fd, &c->hello, sizeof(c->hello)) || c->hello.magic != PROTOCOL_MAGIC)
    {
        fprintf(stderr, "TCP: invalid server hello (rejected? wrong token?)\n");
        close(fd); c->tcp_fd = -1; return false;
    }
    int w = static_cast<int>(c->hello.width), h = static_cast<int>(c->hello.height);
    c->bgra.assign(static_cast<size_t>(w) * h * 4, 0);
    printf("connected over TCP: stream is %dx%d (%s)\n", w, h,
        static_cast<Codec>(c->hello.codec) == Codec::H264 ? "H.264" : "raw");

    std::vector<uint8_t> payload;
    while (!c->quit)
    {
        FrameHeader fh{};
        if (!tcpRecvAll(fd, &fh, sizeof(fh))) { break; }
        payload.resize(fh.size);
        if (fh.size && !tcpRecvAll(fd, payload.data(), fh.size)) { break; }
        handle_message(c, fh.flags, payload.data(), fh.size);
    }
    close(fd);
    c->tcp_fd = -1;
    return true;
}

bool send_packet(QuicClient *c, const uint8_t *data, size_t len)
{
    for (;;)
    {
        ssize_t n = send(c->fd, data, len, 0);
        if (n < 0 && errno == EINTR) { continue; }
        return n >= 0;
    }
}

bool pump_read(QuicClient *c)
{
    uint8_t buf[65536];
    for (;;)
    {
        ssize_t n = recv(c->fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK) { return true; }
            return false;
        }
        if (n == 0) { return true; }
        ngtcp2_path path{};
        path.local.addr     = reinterpret_cast<sockaddr*>(&c->local_addr);
        path.local.addrlen  = c->local_addrlen;
        path.remote.addr    = reinterpret_cast<sockaddr*>(&c->remote_addr);
        path.remote.addrlen = c->remote_addrlen;
        ngtcp2_pkt_info pi{};
        int rv = ngtcp2_conn_read_pkt(c->conn, &path, &pi, buf, static_cast<size_t>(n), now_ns());
        if (rv != 0)
        {
            fprintf(stderr, "ngtcp2_conn_read_pkt: %s\n", ngtcp2_strerror(rv));
            return false;
        }
    }
}

bool pump_write(QuicClient *c)
{
    uint8_t buf[1452];
    ngtcp2_path_storage ps;
    ngtcp2_path_storage_zero(&ps);
    ngtcp2_pkt_info pi{};
    const ngtcp2_tstamp ts = now_ns();

    for (;;)
    {
        if (c->control_stream == -1 && c->handshake_done)
        {
            int64_t sid = -1;
            if (ngtcp2_conn_open_uni_stream(c->conn, &sid, nullptr) == 0) { c->control_stream = sid; }
        }

        int64_t stream_id = -1;
        ngtcp2_vec vec{};
        size_t vcnt = 0;
        if (c->control_stream != -1 && c->ctrl_off < c->ctrl_out.size())
        {
            stream_id = c->control_stream;
            vec.base  = c->ctrl_out.data() + c->ctrl_off;
            vec.len   = c->ctrl_out.size() - c->ctrl_off;
            vcnt = 1;
        }

        ngtcp2_ssize wdatalen = 0;
        ngtcp2_ssize nwrite = ngtcp2_conn_writev_stream(c->conn, &ps.path, &pi, buf, sizeof(buf),
            &wdatalen, NGTCP2_WRITE_STREAM_FLAG_MORE, stream_id, vcnt ? &vec : nullptr, vcnt, ts);

        if (nwrite == NGTCP2_ERR_WRITE_MORE)
        {
            if (wdatalen > 0) { c->ctrl_off += static_cast<size_t>(wdatalen); }
            continue;
        }
        if (nwrite < 0)
        {
            fprintf(stderr, "ngtcp2_conn_writev_stream: %s\n", ngtcp2_strerror(static_cast<int>(nwrite)));
            return false;
        }
        if (wdatalen > 0) { c->ctrl_off += static_cast<size_t>(wdatalen); }
        if (c->ctrl_off > 0 && c->ctrl_off == c->ctrl_out.size()) { c->ctrl_out.clear(); c->ctrl_off = 0; }

        if (nwrite == 0) { return true; }
        if (!send_packet(c, buf, static_cast<size_t>(nwrite))) { return false; }
    }
}

bool client_init(QuicClient *c, const char *host, const char *port)
{
    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host, port, &hints, &res) != 0)
    {
        fprintf(stderr, "could not resolve %s:%s\n", host, port);
        return false;
    }
    c->fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (c->fd < 0 || connect(c->fd, res->ai_addr, res->ai_addrlen) != 0)
    {
        fprintf(stderr, "could not connect UDP to %s:%s\n", host, port);
        freeaddrinfo(res);
        return false;
    }
    std::memcpy(&c->remote_addr, res->ai_addr, res->ai_addrlen);
    c->remote_addrlen = res->ai_addrlen;
    freeaddrinfo(res);
    if (getsockname(c->fd, reinterpret_cast<sockaddr*>(&c->local_addr), &c->local_addrlen) != 0)
    {
        fprintf(stderr, "getsockname failed\n");
        return false;
    }

    if (ngtcp2_crypto_ossl_init() != 0) { fprintf(stderr, "ossl_init failed\n"); return false; }
    c->ssl_ctx = SSL_CTX_new(TLS_client_method());
    if (!c->ssl_ctx) { fprintf(stderr, "SSL_CTX_new failed\n"); return false; }
    SSL_CTX_set_min_proto_version(c->ssl_ctx, TLS1_3_VERSION);
    SSL_CTX_set_max_proto_version(c->ssl_ctx, TLS1_3_VERSION);

    c->ssl = SSL_new(c->ssl_ctx);
    if (!c->ssl) { fprintf(stderr, "SSL_new failed\n"); return false; }
    if (ngtcp2_crypto_ossl_ctx_new(&c->ossl_ctx, c->ssl) != 0)
    {
        fprintf(stderr, "ossl_ctx_new failed\n"); return false;
    }
    if (ngtcp2_crypto_ossl_configure_client_session(c->ssl) != 0)
    {
        fprintf(stderr, "configure_client_session failed\n"); return false;
    }
    c->conn_ref.get_conn = get_conn_cb;
    c->conn_ref.user_data = c;
    SSL_set_app_data(c->ssl, &c->conn_ref);
    SSL_set_connect_state(c->ssl);
    SSL_set_alpn_protos(c->ssl, ALPN, sizeof(ALPN));

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
    cb.rand                     = rand_cb;
    cb.get_new_connection_id2   = get_new_cid_cb;
    cb.handshake_completed      = handshake_completed_cb;
    cb.recv_stream_data         = recv_stream_data_cb;

    ngtcp2_cid dcid{}, scid{};
    dcid.datalen = NGTCP2_MIN_INITIAL_DCIDLEN;
    RAND_bytes(dcid.data, static_cast<int>(dcid.datalen));
    scid.datalen = 8;
    RAND_bytes(scid.data, static_cast<int>(scid.datalen));

    ngtcp2_path path{};
    path.local.addr     = reinterpret_cast<sockaddr*>(&c->local_addr);
    path.local.addrlen  = c->local_addrlen;
    path.remote.addr    = reinterpret_cast<sockaddr*>(&c->remote_addr);
    path.remote.addrlen = c->remote_addrlen;

    ngtcp2_settings settings{};
    ngtcp2_settings_default(&settings);
    settings.initial_ts = now_ns();

    ngtcp2_transport_params params{};
    ngtcp2_transport_params_default(&params);
    params.initial_max_streams_uni     = 3;                // allow server's video stream
    params.initial_max_stream_data_uni = 16 * 1024 * 1024;
    params.initial_max_data            = 64 * 1024 * 1024;

    int rv = ngtcp2_conn_client_new(&c->conn, &dcid, &scid, &path, NGTCP2_PROTO_VER_V1,
        &cb, &settings, &params, nullptr, c);
    if (rv != 0) { fprintf(stderr, "client_new: %s\n", ngtcp2_strerror(rv)); return false; }
    ngtcp2_conn_set_tls_native_handle(c->conn, c->ossl_ctx);
    return true;
}

// The H.264 decoder is shared by both transports, so it is set up once independently of QUIC.
void init_decoder(QuicClient *c)
{
    c->avcodec = avcodec_find_decoder(AV_CODEC_ID_H264);
    c->avctx   = avcodec_alloc_context3(c->avcodec);
    avcodec_open2(c->avctx, c->avcodec, nullptr);
    c->avpkt   = av_packet_alloc();
    c->avframe = av_frame_alloc();
}

// Releases only the QUIC-specific resources (so the decoder can be reused on a TCP fallback).
void free_quic(QuicClient *c)
{
    if (c->conn)     { ngtcp2_conn_del(c->conn); c->conn = nullptr; }
    if (c->ossl_ctx) { ngtcp2_crypto_ossl_ctx_del(c->ossl_ctx); c->ossl_ctx = nullptr; }
    if (c->ssl)      { SSL_set_app_data(c->ssl, nullptr); SSL_free(c->ssl); c->ssl = nullptr; }
    if (c->ssl_ctx)  { SSL_CTX_free(c->ssl_ctx); c->ssl_ctx = nullptr; }
    if (c->fd >= 0)  { close(c->fd); c->fd = -1; }
}

void free_decoder(QuicClient *c)
{
    if (c->sws)     { sws_freeContext(c->sws); }
    if (c->avframe) { av_frame_free(&c->avframe); }
    if (c->avpkt)   { av_packet_free(&c->avpkt); }
    if (c->avctx)   { avcodec_free_context(&c->avctx); }
}

// Runs a QUIC session: connect, authenticate, stream until done. Returns true if a stream was
// established (so the caller need not fall back); false if QUIC never came up (UDP blocked,
// handshake timed out, etc.).
bool run_quic(QuicClient *c, const char *host, const char *port, const std::string& token)
{
    if (!client_init(c, host, port)) { return false; }
    queue_auth(c, token);
    if (!pump_write(c)) { free_quic(c); return false; }

    const uint64_t t0 = now_ns();
    while (!c->quit)
    {
        ngtcp2_tstamp expiry = ngtcp2_conn_get_expiry2(c->conn);
        ngtcp2_tstamp now = now_ns();
        int timeout = 200;
        if (expiry != UINT64_MAX)
        {
            int e = (expiry <= now) ? 0 : static_cast<int>((expiry - now) / NGTCP2_MILLISECONDS);
            if (e < timeout) { timeout = e; }
        }
        pollfd pfd{ c->fd, POLLIN, 0 };
        int pr = poll(&pfd, 1, timeout);
        if (pr < 0) { if (errno == EINTR) { continue; } break; }
        if ((pfd.revents & POLLIN) && !pump_read(c)) { break; }
        process_video(c);
        if (ngtcp2_conn_get_expiry2(c->conn) <= now_ns())
        {
            if (ngtcp2_conn_handle_expiry(c->conn, now_ns()) != 0) { break; }
        }
        if (!pump_write(c)) { break; }

        // If the handshake hasn't completed promptly, QUIC is likely blocked: give up and let
        // the caller fall back to TCP.
        if (!c->handshake_done && (now_ns() - t0) > 3ull * NGTCP2_SECONDS)
        {
            fprintf(stderr, "QUIC handshake timed out\n");
            break;
        }
    }
    for (int i = 0; i < 10; ++i) { pump_write(c); pump_read(c); } // flush final control (Quit)
    bool established = c->got_hello;
    free_quic(c);
    return established;
}

} // namespace

int main(int argc, char *argv[])
{
    const char *host  = (argc >= 2) ? argv[1] : "127.0.0.1";
    const char *port  = (argc >= 3) ? argv[2] : "9000";
    const char *token = (argc >= 4) ? argv[3] : "";
    std::string mode  = (argc >= 5) ? argv[4] : "auto"; // auto | quic | tcp

    QuicClient *c = &g_client;
    ngtcp2_ccerr_default(&c->last_error);
    init_decoder(c);

    // Transport selection: QUIC preferred (auto/quic); on QUIC failure, auto falls back to TCP
    // (the path an ssh -L tunnel or UDP-hostile network needs).
    if (mode == "tcp")
    {
        run_tcp(c, host, port, token);
    }
    else
    {
        bool established = run_quic(c, host, port, token);
        if (!established && mode != "quic")
        {
            printf("QUIC unavailable; falling back to TCP\n");
            run_tcp(c, host, port, token);
        }
    }

    const size_t raw_bytes = static_cast<size_t>(c->hello.width) * c->hello.height * 4
        * static_cast<size_t>(c->received);
    printf("received %d frames (%zu payload bytes), decoded %d frames\n",
        c->received, c->total_encoded, c->decoded);
    if (c->received > 0 && static_cast<Codec>(c->hello.codec) == Codec::H264)
    {
        printf("avg %.1f KB/encoded-frame vs %.1f KB/raw-frame (%.1fx smaller)\n",
            c->total_encoded / 1024.0 / c->received,
            (raw_bytes / static_cast<double>(c->received)) / 1024.0,
            raw_bytes / static_cast<double>(c->total_encoded));
    }

    int decoded = c->decoded;
    free_decoder(c);
    return (decoded > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
