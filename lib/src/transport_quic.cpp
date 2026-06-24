// QUIC transport for interactive remote rendering (step 4).
//
// Preferred transport for direct connections: UDP + TLS + congestion control, with the video
// and control channels mapped to separate QUIC streams. Built on ngtcp2 (the QUIC protocol
// engine) with its OpenSSL crypto binding for the TLS handshake. Only compiled with real
// functionality when MIMIR_HAVE_QUIC is defined (enabled via the MIMIR_ENABLE_QUIC CMake
// option); otherwise listenQuic() reports the missing support and returns nullptr so callers
// transparently fall back to TCP.
//
// Channel mapping:
//   - video   = a server-initiated unidirectional stream carrying Hello + FrameHeader/payload.
//   - control = a client-initiated unidirectional stream carrying fixed-size ControlMsg structs.
// Because QUIC streams are reliable and ordered, the byte framing is identical to the TCP path.
//
// A single I/O thread owns the ngtcp2_conn and the UDP socket and runs the read/write/timer
// pump; the render thread only enqueues video bytes (sendVideo) and drains control (pollControl),
// woken via an eventfd. This keeps all ngtcp2 calls single-threaded, as the library requires.

#include "mimir/transport.hpp"

#include <spdlog/spdlog.h>

namespace mimir::remote
{

#ifdef MIMIR_HAVE_QUIC

} // namespace mimir::remote

#include <ngtcp2/ngtcp2.h>
#include <ngtcp2/ngtcp2_crypto.h>
#include <ngtcp2/ngtcp2_crypto_ossl.h>

// ngtcp2_conn_get_expiry2 was added after 1.22.x; alias it to the original on older releases.
#ifndef HAVE_NGTCP2_CONN_GET_EXPIRY2
#  define ngtcp2_conn_get_expiry2 ngtcp2_conn_get_expiry
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <openssl/x509.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace mimir::remote
{
namespace
{

// QUIC ALPN identifying a mimir stream (length-prefixed, as TLS requires).
const unsigned char ALPN[] = {5, 'm', 'i', 'm', 'i', 'r'};

uint64_t now_ns()
{
    timespec tp{};
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return static_cast<uint64_t>(tp.tv_sec) * NGTCP2_SECONDS + static_cast<uint64_t>(tp.tv_nsec);
}

// Generates an ephemeral self-signed P-256 certificate and installs it on the context. QUIC
// mandates TLS; for the direct-IP deployment a self-signed cert is enough (the client does not
// verify it — encryption without authentication, like an ssh -L tunnel's first connection).
bool installSelfSignedCert(SSL_CTX *ctx)
{
    EVP_PKEY *pkey = EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256");
    if (!pkey) { return false; }
    X509 *x = X509_new();
    bool ok = false;
    if (x)
    {
        ASN1_INTEGER_set(X509_get_serialNumber(x), 1);
        X509_gmtime_adj(X509_getm_notBefore(x), 0);
        X509_gmtime_adj(X509_getm_notAfter(x), 60L * 60 * 24 * 365);
        X509_set_pubkey(x, pkey);
        X509_NAME *name = X509_get_subject_name(x);
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
            reinterpret_cast<const unsigned char*>("mimir"), -1, -1, 0);
        X509_set_issuer_name(x, name);
        if (X509_sign(x, pkey, EVP_sha256()))
        {
            ok = SSL_CTX_use_certificate(ctx, x) == 1 && SSL_CTX_use_PrivateKey(ctx, pkey) == 1;
        }
        X509_free(x);
    }
    EVP_PKEY_free(pkey);
    return ok;
}

int alpnSelectCb(SSL*, const unsigned char **out, unsigned char *outlen,
    const unsigned char *in, unsigned int inlen, void*)
{
    if (SSL_select_next_proto(const_cast<unsigned char**>(out), outlen,
            ALPN, sizeof(ALPN), in, inlen) == OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_OK;
    }
    return SSL_TLSEXT_ERR_ALERT_FATAL;
}

class QuicTransport final : public Transport
{
public:
    QuicTransport() { ngtcp2_ccerr_default(&last_error_); }

    ~QuicTransport() override
    {
        alive_.store(false);
        wake();
        if (io_thread_.joinable()) { io_thread_.join(); }
        if (conn_)    { ngtcp2_conn_del(conn_); }
        if (ossl_ctx_){ ngtcp2_crypto_ossl_ctx_del(ossl_ctx_); }
        if (ssl_)     { SSL_set_app_data(ssl_, nullptr); SSL_free(ssl_); }
        if (ssl_ctx_) { SSL_CTX_free(ssl_ctx_); }
        if (wake_fd_ >= 0) { close(wake_fd_); }
        if (fd_ >= 0) { close(fd_); }
    }

    // --- Transport interface (called from the render thread) ---

    bool sendVideo(const void *data, size_t len) override
    {
        if (!alive_.load()) { return false; }
        {
            std::lock_guard<std::mutex> lock(out_mutex_);
            auto *p = static_cast<const uint8_t*>(data);
            out_buf_.insert(out_buf_.end(), p, p + len);
        }
        wake();
        return true;
    }

    bool pollControl(std::vector<ControlMsg>& out) override
    {
        std::lock_guard<std::mutex> lock(in_mutex_);
        out.insert(out.end(), events_.begin(), events_.end());
        events_.clear();
        return alive_.load();
    }

    // Builds the connection from the client's first datagram and starts the I/O thread. Blocks
    // until the handshake completes (or fails / times out). Returns false on failure.
    bool start(int fd, int wake_fd, const std::string& token,
        const sockaddr_storage& remote, socklen_t remote_len,
        const uint8_t *first_pkt, size_t first_len)
    {
        fd_ = fd;
        wake_fd_ = wake_fd;
        token_ = token;
        remote_addr_ = remote;
        remote_addrlen_ = remote_len;

        // After connect(), the kernel fixes the local address; capture it for the QUIC path.
        local_addrlen_ = sizeof(local_addr_);
        if (getsockname(fd_, reinterpret_cast<sockaddr*>(&local_addr_), &local_addrlen_) != 0)
        {
            spdlog::error("remote(quic): getsockname failed");
            return false;
        }

        ngtcp2_version_cid vc{};
        int rv = ngtcp2_pkt_decode_version_cid(&vc, first_pkt, first_len, NGTCP2_MAX_CIDLEN);
        if (rv != 0)
        {
            spdlog::error("remote(quic): cannot decode client initial: {}", ngtcp2_strerror(rv));
            return false;
        }

        if (!initTls() || !initConn(vc)) { return false; }

        // Feed the client's first packet, then start pumping.
        if (!readPacket(first_pkt, first_len)) { return false; }

        io_thread_ = std::thread([this]{ ioLoop(); });

        // Wait for the TLS handshake AND the client's first control message (the AuthMsg) before
        // declaring the transport ready, so unauthorized clients never get a stream.
        std::unique_lock<std::mutex> lock(handshake_mutex_);
        handshake_cv_.wait_for(lock, std::chrono::seconds(10),
            [this]{ return (handshake_done_.load() && auth_done_.load()) || !alive_.load(); });
        if (!handshake_done_.load())
        {
            spdlog::error("remote(quic): handshake did not complete");
            return false;
        }
        if (!auth_done_.load())
        {
            spdlog::error("remote(quic): client sent no auth message");
            return false;
        }
        if (!auth_ok_.load())
        {
            spdlog::warn("remote(quic): client rejected (bad token)");
            return false;
        }
        spdlog::info("remote(quic): handshake complete, client connected and authenticated");
        return true;
    }

private:
    void wake()
    {
        uint64_t one = 1;
        ssize_t r = write(wake_fd_, &one, sizeof(one));
        (void)r;
    }

    static QuicTransport* self(void *user_data) { return static_cast<QuicTransport*>(user_data); }

    static ngtcp2_conn* getConnCb(ngtcp2_crypto_conn_ref *ref)
    {
        return static_cast<QuicTransport*>(ref->user_data)->conn_;
    }

    static void randCb(uint8_t *dest, size_t destlen, const ngtcp2_rand_ctx*)
    {
        RAND_bytes(dest, static_cast<int>(destlen));
    }

    static int getNewCidCb(ngtcp2_conn*, ngtcp2_cid *cid, ngtcp2_stateless_reset_token *token,
        size_t cidlen, void*)
    {
        if (RAND_bytes(cid->data, static_cast<int>(cidlen)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
        cid->datalen = cidlen;
        if (RAND_bytes(token->data, sizeof(token->data)) != 1) { return NGTCP2_ERR_CALLBACK_FAILURE; }
        return 0;
    }

    static int handshakeCompletedCb(ngtcp2_conn*, void *user_data)
    {
        auto *t = self(user_data);
        {
            std::lock_guard<std::mutex> lock(t->handshake_mutex_);
            t->handshake_done_.store(true);
        }
        t->handshake_cv_.notify_all();
        return 0;
    }

    static int recvStreamDataCb(ngtcp2_conn*, uint32_t, int64_t stream_id, uint64_t,
        const uint8_t *data, size_t datalen, void *user_data, void*)
    {
        return self(user_data)->onControlBytes(stream_id, data, datalen);
    }

    int onControlBytes(int64_t stream_id, const uint8_t *data, size_t datalen)
    {
        bool just_authed = false;
        {
            std::lock_guard<std::mutex> lock(in_mutex_);
            ctrl_buf_.insert(ctrl_buf_.end(), data, data + datalen);

            // The first control-channel message is the AuthMsg; consume and validate it before
            // any ControlMsg parsing.
            if (!auth_done_.load())
            {
                if (ctrl_buf_.size() < sizeof(AuthMsg)) { return passControlCredit(stream_id, datalen); }
                AuthMsg auth{};
                std::memcpy(&auth, ctrl_buf_.data(), sizeof(auth));
                ctrl_buf_.erase(ctrl_buf_.begin(),
                    ctrl_buf_.begin() + static_cast<long>(sizeof(AuthMsg)));
                auth_ok_.store(authOk(auth, token_));
                auth_done_.store(true);
                just_authed = true;
            }

            // The rest of the control channel is a stream of fixed-size ControlMsg structs.
            while (ctrl_buf_.size() >= sizeof(ControlMsg))
            {
                ControlMsg msg{};
                std::memcpy(&msg, ctrl_buf_.data(), sizeof(msg));
                ctrl_buf_.erase(ctrl_buf_.begin(),
                    ctrl_buf_.begin() + static_cast<long>(sizeof(ControlMsg)));
                events_.push_back(msg);
            }
        }
        if (just_authed) { handshake_cv_.notify_all(); }
        return passControlCredit(stream_id, datalen);
    }

    // Return consumed bytes to the flow-control window so the client can keep sending.
    int passControlCredit(int64_t stream_id, size_t datalen)
    {
        ngtcp2_conn_extend_max_stream_offset(conn_, stream_id, datalen);
        ngtcp2_conn_extend_max_offset(conn_, datalen);
        return 0;
    }

    bool initTls()
    {
        static const int init_once = ngtcp2_crypto_ossl_init();
        if (init_once != 0) { spdlog::error("remote(quic): ngtcp2_crypto_ossl_init failed"); return false; }

        ssl_ctx_ = SSL_CTX_new(TLS_server_method());
        if (!ssl_ctx_) { spdlog::error("remote(quic): SSL_CTX_new failed"); return false; }
        SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_3_VERSION);
        SSL_CTX_set_max_proto_version(ssl_ctx_, TLS1_3_VERSION);
        SSL_CTX_set_alpn_select_cb(ssl_ctx_, alpnSelectCb, nullptr);
        if (!installSelfSignedCert(ssl_ctx_))
        {
            spdlog::error("remote(quic): failed to install self-signed certificate");
            return false;
        }

        ssl_ = SSL_new(ssl_ctx_);
        if (!ssl_) { spdlog::error("remote(quic): SSL_new failed"); return false; }
        if (ngtcp2_crypto_ossl_ctx_new(&ossl_ctx_, ssl_) != 0)
        {
            spdlog::error("remote(quic): ngtcp2_crypto_ossl_ctx_new failed");
            return false;
        }
        if (ngtcp2_crypto_ossl_configure_server_session(ssl_) != 0)
        {
            spdlog::error("remote(quic): configure_server_session failed");
            return false;
        }
        conn_ref_.get_conn = getConnCb;
        conn_ref_.user_data = this;
        SSL_set_app_data(ssl_, &conn_ref_);
        SSL_set_accept_state(ssl_);
        return true;
    }

    bool initConn(const ngtcp2_version_cid& vc)
    {
        ngtcp2_path path{};
        path.local.addr     = reinterpret_cast<sockaddr*>(&local_addr_);
        path.local.addrlen  = local_addrlen_;
        path.remote.addr    = reinterpret_cast<sockaddr*>(&remote_addr_);
        path.remote.addrlen = remote_addrlen_;

        ngtcp2_callbacks cb{};
        cb.recv_client_initial      = ngtcp2_crypto_recv_client_initial_cb;
        cb.recv_crypto_data         = ngtcp2_crypto_recv_crypto_data_cb;
        cb.encrypt                  = ngtcp2_crypto_encrypt_cb;
        cb.decrypt                  = ngtcp2_crypto_decrypt_cb;
        cb.hp_mask                  = ngtcp2_crypto_hp_mask_cb;
        cb.update_key               = ngtcp2_crypto_update_key_cb;
        cb.delete_crypto_aead_ctx   = ngtcp2_crypto_delete_crypto_aead_ctx_cb;
        cb.delete_crypto_cipher_ctx = ngtcp2_crypto_delete_crypto_cipher_ctx_cb;
        cb.get_path_challenge_data2 = ngtcp2_crypto_get_path_challenge_data2_cb;
        cb.version_negotiation      = ngtcp2_crypto_version_negotiation_cb;
        cb.rand                     = randCb;
        cb.get_new_connection_id2   = getNewCidCb;
        cb.handshake_completed      = handshakeCompletedCb;
        cb.recv_stream_data         = recvStreamDataCb;

        ngtcp2_cid scid{};
        scid.datalen = 18;
        RAND_bytes(scid.data, static_cast<int>(scid.datalen));

        ngtcp2_cid dcid{};                 // server's DCID = client's SCID
        dcid.datalen = vc.scidlen;
        std::memcpy(dcid.data, vc.scid, vc.scidlen);

        ngtcp2_settings settings{};
        ngtcp2_settings_default(&settings);
        settings.initial_ts = now_ns();

        ngtcp2_transport_params params{};
        ngtcp2_transport_params_default(&params);
        params.initial_max_streams_uni        = 3;                  // client may open control
        params.initial_max_streams_bidi       = 0;
        params.initial_max_stream_data_uni    = 16 * 1024 * 1024;   // ample for video frames
        params.initial_max_data               = 64 * 1024 * 1024;
        params.original_dcid_present           = 1;
        params.original_dcid.datalen = vc.dcidlen;                  // DCID from client's Initial
        std::memcpy(params.original_dcid.data, vc.dcid, vc.dcidlen);

        int rv = ngtcp2_conn_server_new(&conn_, &dcid, &scid, &path, vc.version,
            &cb, &settings, &params, nullptr, this);
        if (rv != 0)
        {
            spdlog::error("remote(quic): ngtcp2_conn_server_new: {}", ngtcp2_strerror(rv));
            return false;
        }
        ngtcp2_conn_set_tls_native_handle(conn_, ossl_ctx_);
        return true;
    }

    bool readPacket(const uint8_t *data, size_t len)
    {
        ngtcp2_path path{};
        path.local.addr     = reinterpret_cast<sockaddr*>(&local_addr_);
        path.local.addrlen  = local_addrlen_;
        path.remote.addr    = reinterpret_cast<sockaddr*>(&remote_addr_);
        path.remote.addrlen = remote_addrlen_;
        ngtcp2_pkt_info pi{};
        int rv = ngtcp2_conn_read_pkt(conn_, &path, &pi, data, len, now_ns());
        if (rv != 0)
        {
            spdlog::warn("remote(quic): read_pkt: {}", ngtcp2_strerror(rv));
            return false;
        }
        return true;
    }

    // Drains incoming datagrams on the connected UDP socket into ngtcp2.
    bool pumpRead()
    {
        uint8_t buf[65536];
        for (;;)
        {
            ssize_t n = recv(fd_, buf, sizeof(buf), MSG_DONTWAIT);
            if (n < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK) { return true; }
                return false;
            }
            if (n == 0) { return true; }
            if (!readPacket(buf, static_cast<size_t>(n))) { return false; }
        }
    }

    // Produces and sends outgoing QUIC datagrams: queued video bytes plus protocol packets
    // (ACKs, handshake, flow control). Lazily opens the video stream once handshake credit
    // allows. Returns false on a fatal error.
    bool pumpWrite()
    {
        uint8_t buf[1452];
        ngtcp2_path_storage ps;
        ngtcp2_path_storage_zero(&ps);
        ngtcp2_pkt_info pi{};
        const ngtcp2_tstamp ts = now_ns();

        for (;;)
        {
            if (video_stream_ == -1 && handshake_done_.load())
            {
                int64_t sid = -1;
                if (ngtcp2_conn_open_uni_stream(conn_, &sid, nullptr) == 0) { video_stream_ = sid; }
            }

            int64_t stream_id = -1;
            ngtcp2_vec vec{};
            size_t vcnt = 0;
            std::unique_lock<std::mutex> lock(out_mutex_);
            if (video_stream_ != -1 && out_off_ < out_buf_.size())
            {
                stream_id = video_stream_;
                vec.base  = out_buf_.data() + out_off_;
                vec.len   = out_buf_.size() - out_off_;
                vcnt = 1;
            }

            ngtcp2_ssize wdatalen = 0;
            ngtcp2_ssize nwrite = ngtcp2_conn_writev_stream(conn_, &ps.path, &pi, buf, sizeof(buf),
                &wdatalen, NGTCP2_WRITE_STREAM_FLAG_MORE, stream_id, vcnt ? &vec : nullptr, vcnt, ts);

            if (nwrite == NGTCP2_ERR_WRITE_MORE)
            {
                if (wdatalen > 0) { out_off_ += static_cast<size_t>(wdatalen); }
                lock.unlock();
                continue;
            }
            if (nwrite < 0)
            {
                lock.unlock();
                spdlog::error("remote(quic): writev_stream: {}", ngtcp2_strerror(static_cast<int>(nwrite)));
                return false;
            }
            if (wdatalen > 0) { out_off_ += static_cast<size_t>(wdatalen); }
            if (out_off_ > 0 && out_off_ == out_buf_.size()) { out_buf_.clear(); out_off_ = 0; }
            lock.unlock();

            if (nwrite == 0) { return true; } // nothing left to send right now
            if (send(fd_, buf, static_cast<size_t>(nwrite), 0) < 0) { return false; }
        }
    }

    void ioLoop()
    {
        if (!pumpWrite()) { fail(); return; }

        while (alive_.load())
        {
            ngtcp2_tstamp expiry = ngtcp2_conn_get_expiry2(conn_);
            ngtcp2_tstamp now = now_ns();
            int timeout = -1;
            if (expiry != UINT64_MAX)
            {
                timeout = (expiry <= now) ? 0 : static_cast<int>((expiry - now) / NGTCP2_MILLISECONDS);
            }

            pollfd pfds[2] = {
                { fd_,      POLLIN, 0 },
                { wake_fd_, POLLIN, 0 },
            };
            int pr = poll(pfds, 2, timeout);
            if (pr < 0) { if (errno == EINTR) { continue; } break; }

            if (pfds[1].revents & POLLIN)
            {
                uint64_t drain = 0;
                ssize_t r = read(wake_fd_, &drain, sizeof(drain));
                (void)r; // just a wakeup; the work is in out_buf_ / the alive flag
            }
            if ((pfds[0].revents & POLLIN) && !pumpRead()) { break; }

            if (ngtcp2_conn_get_expiry2(conn_) <= now_ns())
            {
                if (ngtcp2_conn_handle_expiry(conn_, now_ns()) != 0) { break; }
            }
            if (!pumpWrite()) { break; }
        }
        fail();
    }

    // Marks the session dead and wakes any handshake waiter (used on any loop-ending error).
    void fail()
    {
        alive_.store(false);
        handshake_cv_.notify_all();
    }

    // Connection state (owned by the I/O thread after start()).
    int fd_ = -1;
    int wake_fd_ = -1;
    sockaddr_storage local_addr_{};
    sockaddr_storage remote_addr_{};
    socklen_t local_addrlen_ = 0;
    socklen_t remote_addrlen_ = 0;

    SSL_CTX *ssl_ctx_ = nullptr;
    SSL *ssl_ = nullptr;
    ngtcp2_crypto_ossl_ctx *ossl_ctx_ = nullptr;
    ngtcp2_conn *conn_ = nullptr;
    ngtcp2_crypto_conn_ref conn_ref_{};
    ngtcp2_ccerr last_error_{};

    int64_t video_stream_ = -1;
    std::string token_;                    // expected auth token (empty = accept any)

    std::atomic<bool> alive_{true};
    std::atomic<bool> handshake_done_{false};
    std::atomic<bool> auth_done_{false};
    std::atomic<bool> auth_ok_{false};
    std::mutex handshake_mutex_;
    std::condition_variable handshake_cv_;

    std::mutex out_mutex_;
    std::vector<uint8_t> out_buf_;
    size_t out_off_ = 0;

    std::mutex in_mutex_;
    std::vector<uint8_t> ctrl_buf_;
    std::deque<ControlMsg> events_;

    std::thread io_thread_;
};

} // namespace

namespace
{
// Binds a fresh UDP socket to the given port (INADDR_ANY) with address/port reuse so a listener
// and per-session connected socket can share it. Returns -1 on failure.
int bindUdp(uint16_t port)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) { return -1; }
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) { close(fd); return -1; }
    return fd;
}
} // namespace

std::unique_ptr<Transport> listenQuic(uint16_t port, const std::string& token)
{
    // A persistent unconnected "listener" socket waits for the next client's QUIC Initial and
    // cheaply drops stray packets (e.g. retransmits from a just-rejected client are short-header
    // 1-RTT packets, which decode_version_cid rejects) without spinning up a connection. Only the
    // listener bind is fatal; failed/unauthorized clients just leave us waiting for the next.
    int listen_fd = bindUdp(port);
    if (listen_fd < 0) { spdlog::error("remote(quic): failed to bind UDP port {}", port); return nullptr; }
    spdlog::info("remote(quic): waiting for a client on UDP port {}", port);

    for (;;)
    {
        uint8_t first[65536];
        sockaddr_storage remote{};
        socklen_t remote_len = sizeof(remote);
        ssize_t n = recvfrom(listen_fd, first, sizeof(first), 0,
            reinterpret_cast<sockaddr*>(&remote), &remote_len);
        if (n <= 0) { continue; }

        // Only a genuine QUIC Initial starts a session; anything else (short-header 1-RTT or
        // long-header Handshake retransmits from a just-rejected client, etc.) is dropped
        // without cost, so a departing/unauthorized client can't make us churn connections.
        ngtcp2_version_cid vc{};
        if (ngtcp2_pkt_decode_version_cid(&vc, first, static_cast<size_t>(n), NGTCP2_MAX_CIDLEN) != 0)
        {
            continue;
        }
        ngtcp2_pkt_hd hd{};
        if (ngtcp2_pkt_decode_hd_long(&hd, first, static_cast<size_t>(n)) < 0
            || hd.type != NGTCP2_PKT_INITIAL)
        {
            continue;
        }

        // Per-session socket on the same port, connected to this peer (so the kernel routes the
        // peer's subsequent datagrams here, ahead of the wildcard listener).
        int sfd = bindUdp(port);
        if (sfd < 0) { spdlog::error("remote(quic): failed to create session socket"); continue; }
        if (connect(sfd, reinterpret_cast<sockaddr*>(&remote), remote_len) != 0)
        {
            spdlog::error("remote(quic): connect to client failed"); close(sfd); continue;
        }
        int wake_fd = eventfd(0, EFD_NONBLOCK);
        if (wake_fd < 0) { spdlog::error("remote(quic): eventfd failed"); close(sfd); continue; }

        auto transport = std::make_unique<QuicTransport>();
        if (transport->start(sfd, wake_fd, token, remote, remote_len, first, static_cast<size_t>(n)))
        {
            close(listen_fd); // session owns sfd; the listener is no longer needed
            return transport;
        }
        // Handshake/auth failed; transport dtor closed sfd/wake_fd. Keep waiting on the listener.
        spdlog::warn("remote(quic): client setup/auth failed, waiting for the next");
    }
}

#else // !MIMIR_HAVE_QUIC

std::unique_ptr<Transport> listenQuic(uint16_t, const std::string&)
{
    spdlog::error("remote(quic): library built without QUIC support; "
        "rebuild with -DMIMIR_ENABLE_QUIC=ON (needs ngtcp2), or use the TCP transport");
    return nullptr;
}

#endif // MIMIR_HAVE_QUIC

} // namespace mimir::remote
