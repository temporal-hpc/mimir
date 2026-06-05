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

namespace mimir
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

} // namespace

void MimirInstance::serveRemote(uint16_t port, std::function<void(void)> compute, size_t max_iters)
{
    prepare();

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
        .width  = swapchain.extent.width,
        .height = swapchain.extent.height,
        .format = static_cast<uint32_t>(remote::PixelFormat::BGRA8),
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
        remote::FrameHeader header{ .size = static_cast<uint32_t>(frame.size()) };
        if (!sendAll(client, &header, sizeof(header)) ||
            !sendAll(client, frame.data(), frame.size()))
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
