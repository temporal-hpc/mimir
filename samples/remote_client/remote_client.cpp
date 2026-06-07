// Remote rendering client sample (step 2): a thin, dependency-free native client.
//
// Connects to run_remote_server over TCP, receives raw streamed frames, and exercises the
// control round-trip. To keep the bring-up verifiable without a display, this client saves a
// few frames to PPM rather than opening a window: it pauses the server-side simulation, saves a
// frame, sends a camera rotation, and saves another frame — so the two PPMs differ only because
// the control event changed the view. A windowed blit client is the next increment.
//
// Run from build/samples/:  ./run_remote_client [host] [port]

#include <mimir/remote_protocol.hpp>
using namespace mimir::remote;

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool sendAll(int fd, const void *buf, size_t len)
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

static bool recvAll(int fd, void *buf, size_t len)
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

static void savePpm(const std::string& path, const std::vector<unsigned char>& bgra,
    uint32_t w, uint32_t h)
{
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << w << " " << h << "\n255\n";
    for (uint32_t i = 0; i < w * h; ++i)
    {
        f.put(static_cast<char>(bgra[i * 4 + 2])); // R
        f.put(static_cast<char>(bgra[i * 4 + 1])); // G
        f.put(static_cast<char>(bgra[i * 4 + 0])); // B
    }
    printf("  saved %s\n", path.c_str());
}

static bool sendControl(int fd, ControlKind kind, float a = 0.f, float b = 0.f)
{
    ControlMsg msg{};
    msg.kind = static_cast<uint8_t>(kind);
    msg.a = a;
    msg.b = b;
    return sendAll(fd, &msg, sizeof(msg));
}

// Sends the auth handshake (always first on the control channel; token may be empty).
static bool sendAuth(int fd, const std::string& token)
{
    AuthMsg a{};
    a.magic = AUTH_MAGIC;
    size_t n = token.size() < TOKEN_MAX ? token.size() : static_cast<size_t>(TOKEN_MAX);
    std::memcpy(a.token, token.data(), n);
    return sendAll(fd, &a, sizeof(a));
}

int main(int argc, char *argv[])
{
    std::string host  = (argc >= 2)? argv[1] : "127.0.0.1";
    std::string port  = (argc >= 3)? argv[2] : "9000";
    std::string token = (argc >= 4)? argv[3] : "";

    // Resolve and connect.
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
    uint32_t w = hello.width, h = hello.height;
    if (static_cast<Codec>(hello.codec) != Codec::RawBGRA)
    {
        fprintf(stderr, "server is streaming codec %u, but this client only decodes raw "
            "frames; use run_remote_decode for H.264\n", hello.codec);
        close(fd);
        return EXIT_FAILURE;
    }
    printf("connected: stream is %ux%u (raw)\n", w, h);

    std::vector<unsigned char> frame;
    int count = 0;
    while (true)
    {
        FrameHeader header{};
        if (!recvAll(fd, &header, sizeof(header))) { break; }
        frame.resize(header.size);
        if (!recvAll(fd, frame.data(), header.size)) { break; }
        if (header.flags & (FRAME_STATS | FRAME_HELLO)) { continue; } // not a frame
        count++;

        // Scripted interaction to verify the control round-trip:
        if (count == 1)  { savePpm("client_frame0.ppm", frame, w, h); }
        if (count == 3)  { printf("pausing simulation\n");  sendControl(fd, ControlKind::TogglePause); }
        if (count == 6)  { savePpm("client_pre_rotate.ppm", frame, w, h); }
        if (count == 7)  { printf("sending camera rotate\n"); sendControl(fd, ControlKind::CameraRotate, 150.f, 40.f); }
        if (count == 15)
        {
            savePpm("client_post_rotate.ppm", frame, w, h);
            printf("sending quit\n");
            sendControl(fd, ControlKind::Quit);
            break;
        }
    }

    printf("received %d frames\n", count);
    close(fd);
    return (count > 0)? EXIT_SUCCESS : EXIT_FAILURE;
}
