// TCP transport for interactive remote rendering (step 4).
//
// Both logical channels share one TCP connection: the server writes the video channel (Hello +
// length-prefixed frames) on the render thread, while a background receiver thread reads the
// fixed-size ControlMsg structs the client sends back and queues them for the render thread to
// drain. This is the everywhere-works fallback (also what an `ssh -L` tunnel carries), accepting
// TCP head-of-line blocking as the cost. QUIC is preferred for direct connections (transport_quic).

#include "mimir/transport.hpp"

#include <spdlog/spdlog.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

namespace mimir::remote
{

namespace
{

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

class TcpTransport final : public Transport
{
public:
    TcpTransport(int client_fd, std::string peer)
        : client_fd_(client_fd), peer_(std::move(peer))
    {
        receiver_ = std::thread([this]
        {
            ControlMsg msg{};
            while (alive_.load())
            {
                if (!recvAll(client_fd_, &msg, sizeof(msg))) { alive_.store(false); break; }
                std::lock_guard<std::mutex> lock(mutex_);
                events_.push_back(msg);
            }
        });
    }

    ~TcpTransport() override
    {
        alive_.store(false);
        shutdown(client_fd_, SHUT_RDWR);
        if (receiver_.joinable()) { receiver_.join(); }
        close(client_fd_);
        // The listening socket outlives the session: it belongs to the TcpListener, which keeps
        // accepting the next client while the server's simulation continues.
    }

    bool sendVideo(const void *data, size_t len) override
    {
        if (!sendAll(client_fd_, data, len)) { alive_.store(false); return false; }
        return true;
    }

    bool pollControl(std::vector<ControlMsg>& out) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        out.insert(out.end(), events_.begin(), events_.end());
        events_.clear();
        return alive_.load();
    }

    std::string peerName() const override { return peer_; }

private:
    int client_fd_ = -1;
    std::string peer_;
    std::atomic<bool> alive_{true};
    std::mutex mutex_;
    std::deque<ControlMsg> events_;
    std::thread receiver_;
};

// Persistent accept socket, polled non-blockingly from the server's simulation loop.
class TcpListener final : public Listener
{
public:
    TcpListener(int listen_fd, std::string token) : listen_fd_(listen_fd), token_(std::move(token)) {}
    ~TcpListener() override { close(listen_fd_); }

    std::unique_ptr<Transport> poll() override
    {
        pollfd pfd{ listen_fd_, POLLIN, 0 };
        if (::poll(&pfd, 1, 0) <= 0 || !(pfd.revents & POLLIN)) { return nullptr; }
        int client = accept(listen_fd_, nullptr, nullptr);
        if (client < 0) { return nullptr; }
        int one = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); // low latency

        // The client sends an AuthMsg as the very first control-channel bytes; validate before
        // accepting the session so unauthorized clients never get a stream. The read is bounded
        // so a stalling client can only briefly pause the simulation loop, not hang it.
        timeval auth_timeout{ 2, 0 };
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &auth_timeout, sizeof(auth_timeout));
        AuthMsg auth{};
        if (!recvAll(client, &auth, sizeof(auth)) || !authOk(auth, token_))
        {
            spdlog::warn("remote(tcp): client rejected (bad token or handshake)");
            shutdown(client, SHUT_RDWR);
            close(client);
            return nullptr; // keep listening; the next poll may find a valid client
        }
        timeval no_timeout{ 0, 0 };
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &no_timeout, sizeof(no_timeout));
        char peer[HOST_MAX + 1] = {}; std::memcpy(peer, auth.client, HOST_MAX);
        spdlog::info("remote(tcp): client connected and authenticated");
        return std::make_unique<TcpTransport>(client, peer);
    }

private:
    int listen_fd_ = -1;
    std::string token_;
};

} // namespace

std::unique_ptr<Listener> makeTcpListener(uint16_t port, const std::string& token)
{
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { spdlog::error("remote(tcp): failed to create socket"); return nullptr; }
    int yes = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(port);
    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        spdlog::error("remote(tcp): failed to bind port {}", port);
        close(listen_fd);
        return nullptr;
    }
    listen(listen_fd, 1);
    spdlog::info("remote(tcp): listening on port {}", port);
    return std::make_unique<TcpListener>(listen_fd, token);
}

} // namespace mimir::remote
