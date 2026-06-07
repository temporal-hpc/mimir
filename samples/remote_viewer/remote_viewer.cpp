// Windowed remote rendering client (step 2.5): a real interactive thin client.
//
// Connects to run_remote_server, displays the streamed raw frames in a window, and sends the
// mouse/keyboard interaction back as control events — so dragging here rotates/pans/zooms the
// camera on the server. Depends only on GLFW + OpenGL + the wire-protocol header (no mimir,
// CUDA, or Vulkan), modelling the thin native client.
//
// Controls: left-drag rotate, right-drag zoom, middle-drag pan, P pause, Q/Esc quit.
// Run from build/samples/:  ./run_remote_viewer [host] [port]

#include <mimir/remote_protocol.hpp>
using namespace mimir::remote;

#include <GLFW/glfw3.h>
#include <GL/gl.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{

int g_fd = -1;

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

void sendControl(ControlKind kind, float a = 0.f, float b = 0.f)
{
    if (g_fd < 0) { return; }
    ControlMsg msg{};
    msg.kind = static_cast<uint8_t>(kind);
    msg.a = a;
    msg.b = b;
    sendAll(g_fd, &msg, sizeof(msg));
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

// Mouse drag state, mirrors the engine's local GLFW input mapping.
struct Input { double last_x = 0, last_y = 0; bool left = false, right = false, middle = false; };
Input g_input;

void cursorCallback(GLFWwindow*, double x, double y)
{
    float dx = static_cast<float>(g_input.last_x - x);
    float dy = static_cast<float>(g_input.last_y - y);
    g_input.last_x = x;
    g_input.last_y = y;
    if (g_input.left)   { sendControl(ControlKind::CameraRotate, dx, dy); }
    if (g_input.right)  { sendControl(ControlKind::CameraZoom, dy); }
    if (g_input.middle) { sendControl(ControlKind::CameraPan, dx, dy); }
}

void buttonCallback(GLFWwindow*, int button, int action, int)
{
    bool pressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)   { g_input.left = pressed; }
    if (button == GLFW_MOUSE_BUTTON_RIGHT)  { g_input.right = pressed; }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) { g_input.middle = pressed; }
}

void keyCallback(GLFWwindow *win, int key, int, int action, int)
{
    if (action != GLFW_PRESS) { return; }
    if (key == GLFW_KEY_P) { sendControl(ControlKind::TogglePause); }
    if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) { glfwSetWindowShouldClose(win, GLFW_TRUE); }
}

} // namespace

int main(int argc, char *argv[])
{
    std::string host  = (argc >= 2)? argv[1] : "127.0.0.1";
    std::string port  = (argc >= 3)? argv[2] : "9000";
    std::string token = (argc >= 4)? argv[3] : "";

    // Connect to the server.
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0)
    {
        fprintf(stderr, "could not resolve %s:%s\n", host.c_str(), port.c_str());
        return EXIT_FAILURE;
    }
    g_fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (g_fd < 0 || connect(g_fd, res->ai_addr, res->ai_addrlen) < 0)
    {
        fprintf(stderr, "could not connect to %s:%s\n", host.c_str(), port.c_str());
        return EXIT_FAILURE;
    }
    freeaddrinfo(res);

    // The server reads our AuthMsg before sending anything, so send it first.
    if (!sendAuth(g_fd, token)) { fprintf(stderr, "failed to send auth\n"); return EXIT_FAILURE; }

    Hello hello{};
    if (!recvAll(g_fd, &hello, sizeof(hello)) || hello.magic != PROTOCOL_MAGIC)
    {
        fprintf(stderr, "invalid server hello (rejected? wrong token?)\n");
        return EXIT_FAILURE;
    }
    if (static_cast<Codec>(hello.codec) != Codec::RawBGRA)
    {
        fprintf(stderr, "server is streaming codec %u, but this viewer only decodes raw "
            "frames; use run_remote_decode for H.264\n", hello.codec);
        close(g_fd);
        return EXIT_FAILURE;
    }
    int w = static_cast<int>(hello.width), h = static_cast<int>(hello.height);
    printf("connected: stream is %dx%d (raw)\n", w, h);

    // A background thread receives frames into a shared buffer so the UI stays responsive.
    std::mutex frame_mutex;
    std::vector<unsigned char> latest(static_cast<size_t>(w) * h * 4, 0);
    std::atomic<bool> running{true};
    std::thread receiver([&]
    {
        std::vector<unsigned char> buf;
        while (running.load())
        {
            FrameHeader header{};
            if (!recvAll(g_fd, &header, sizeof(header))) { break; }
            buf.resize(header.size);
            if (!recvAll(g_fd, buf.data(), header.size)) { break; }
            if (header.flags & (FRAME_STATS | FRAME_HELLO)) { continue; } // not a frame
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest.swap(buf);
        }
        running.store(false);
    });

    // Window + GL setup (legacy fixed-function: upload the frame with glDrawPixels).
    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return EXIT_FAILURE; }
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(w, h, "mimir remote viewer", nullptr, nullptr);
    if (!window) { fprintf(stderr, "window creation failed\n"); glfwTerminate(); return EXIT_FAILURE; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetMouseButtonCallback(window, buttonCallback);
    glfwSetKeyCallback(window, keyCallback);

    std::vector<unsigned char> display(static_cast<size_t>(w) * h * 4);
    while (!glfwWindowShouldClose(window) && running.load())
    {
        glfwPollEvents();
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            std::memcpy(display.data(), latest.data(), display.size());
        }

        int fbw, fbh;
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClear(GL_COLOR_BUFFER_BIT);
        // Frames are top-row-first (Vulkan); flip vertically with a negative pixel zoom.
        glPixelZoom(static_cast<float>(fbw) / w, -static_cast<float>(fbh) / h);
        glRasterPos2f(-1.f, 1.f);
        glDrawPixels(w, h, GL_BGRA, GL_UNSIGNED_BYTE, display.data());
        glfwSwapBuffers(window);
    }

    running.store(false);
    sendControl(ControlKind::Quit);
    shutdown(g_fd, SHUT_RDWR);
    if (receiver.joinable()) { receiver.join(); }
    close(g_fd);
    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
