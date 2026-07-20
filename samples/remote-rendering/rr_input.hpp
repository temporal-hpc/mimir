// Reusable input capture for mimir remote-rendering clients.
//
// Translates local GLFW mouse/keyboard into camera ControlKind events for a headless mimir server,
// so a thin client doesn't hand-roll the trackball-vs-fly input mapping. Depends ONLY on GLFW and the
// wire protocol (mimir/remote_protocol.hpp) -- no mimir engine, CUDA or Vulkan -- so any remote client
// can share it. Header-only.
//
// The client stays in control of transport and UI: it supplies how an event is sent (emit), whether
// the server runs a fly camera (is_fly), and optional hooks for the non-camera keys. Usage:
//
//   mimir::rr::InputCapture input;
//   input.emit          = [](ControlKind k, float a, float b) { ui_control(k, a, b); };
//   input.is_fly        = [] { return g_fly.load(std::memory_order_relaxed); };
//   input.on_toggle_hud = [] { g_hud.visible.store(!g_hud.visible.load()); };   // 'H' (optional)
//   input.on_quit       = [window] { glfwSetWindowShouldClose(window, GLFW_TRUE); }; // optional
//   input.install(window);
//   while (...) { glfwPollEvents(); input.pollMovement(window); /* ... */ }
#pragma once

#include <GLFW/glfw3.h>
#include <mimir/remote_protocol.hpp> // ControlKind

#include <functional>

namespace mimir::rr
{

using remote::ControlKind; // the wire-protocol event kind (mimir/remote_protocol.hpp)

// Maps GLFW input to camera control events. Install once on a window (it claims the window's user
// pointer for callback dispatch), then call pollMovement() once per frame for held-key movement.
struct InputCapture
{
    // Sends a control event to the server: event kind plus up to two float args. Required.
    std::function<void(ControlKind, float, float)> emit;
    // Whether the server runs a Fly camera. Fly maps left-drag -> mouse-look (CameraLook) and WASD ->
    // CameraMove; a trackball maps left/right/middle drag -> orbit/zoom/pan. Defaults to trackball.
    std::function<bool()> is_fly = [] { return false; };
    // Optional client-UI hooks. If on_toggle_pause is unset, 'P' is sent as ControlKind::TogglePause;
    // if on_quit is unset, Q/Esc/Ctrl+W call glfwSetWindowShouldClose.
    std::function<void()> on_toggle_hud   = {}; // 'H'
    std::function<void()> on_toggle_pause = {}; // 'P'
    std::function<void()> on_quit         = {}; // Q / Esc / Ctrl+W

    // Drag state (internal; public only so the struct stays an aggregate).
    double last_x = 0.0, last_y = 0.0;
    bool left = false, right = false, middle = false;

    // Registers the cursor/button/key callbacks on `window` and claims its user pointer.
    void install(GLFWwindow* window)
    {
        glfwSetWindowUserPointer(window, this);
        glfwSetCursorPosCallback(window, &InputCapture::cursorCb);
        glfwSetMouseButtonCallback(window, &InputCapture::buttonCb);
        glfwSetKeyCallback(window, &InputCapture::keyCb);
    }

    // Once per frame: held WASD -> continuous CameraMove (fly only; trackball servers ignore it).
    // Forward follows the gaze on the server, so look-up + W climbs.
    void pollMovement(GLFWwindow* window) const
    {
        if (!emit || !is_fly || !is_fly()) { return; }
        float strafe  = (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS ? 1.f : 0.f)
                      - (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS ? 1.f : 0.f);
        float forward = (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ? 1.f : 0.f)
                      - (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ? 1.f : 0.f);
        if (strafe != 0.f || forward != 0.f) { emit(ControlKind::CameraMove, strafe, forward); }
    }

private:
    static InputCapture* self(GLFWwindow* w)
    {
        return static_cast<InputCapture*>(glfwGetWindowUserPointer(w));
    }

    static void cursorCb(GLFWwindow* w, double x, double y)
    {
        auto* in = self(w);
        if (in == nullptr) { return; }
        float dx = static_cast<float>(in->last_x - x);
        float dy = static_cast<float>(in->last_y - y);
        in->last_x = x; in->last_y = y;
        if (!in->emit) { return; }
        if (in->is_fly && in->is_fly())
        {
            if (in->left) { in->emit(ControlKind::CameraLook, dx, dy); } // left-drag = mouse-look
            return;
        }
        if (in->left)   { in->emit(ControlKind::CameraRotate, dx, dy); }
        if (in->right)  { in->emit(ControlKind::CameraZoom, dy, 0.f); }
        if (in->middle) { in->emit(ControlKind::CameraPan, dx, dy); }
    }

    static void buttonCb(GLFWwindow* w, int button, int action, int /*mods*/)
    {
        auto* in = self(w);
        if (in == nullptr) { return; }
        bool pressed = (action == GLFW_PRESS);
        if (button == GLFW_MOUSE_BUTTON_LEFT)   { in->left = pressed; }
        if (button == GLFW_MOUSE_BUTTON_RIGHT)  { in->right = pressed; }
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) { in->middle = pressed; }
    }

    static void keyCb(GLFWwindow* w, int key, int /*scancode*/, int action, int mods)
    {
        auto* in = self(w);
        if (in == nullptr || action != GLFW_PRESS) { return; }
        if (key == GLFW_KEY_P)
        {
            if (in->on_toggle_pause) { in->on_toggle_pause(); }
            else if (in->emit)       { in->emit(ControlKind::TogglePause, 0.f, 0.f); }
        }
        if (key == GLFW_KEY_H && in->on_toggle_hud) { in->on_toggle_hud(); }
        const bool ctrl_w = (key == GLFW_KEY_W) && (mods & GLFW_MOD_CONTROL);
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE || ctrl_w)
        {
            if (in->on_quit) { in->on_quit(); }
            else { glfwSetWindowShouldClose(w, GLFW_TRUE); }
        }
    }
};

} // namespace mimir::rr
