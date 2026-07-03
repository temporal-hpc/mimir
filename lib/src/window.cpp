#include "mimir/window.hpp"

#include <imgui.h>

#include "mimir/engine.hpp"
#include "mimir/camera.hpp"
#include "mimir/validation.hpp"

#include <algorithm> // std::clamp

namespace mimir::validation
{

// Converts GLFW result codes into strings
const char *getGlfwErrorString(int code)
{
    switch (code)
    {
#define STR(r) case GLFW_ ##r: return #r
        STR(NOT_INITIALIZED);
        STR(NO_CURRENT_CONTEXT);
        STR(INVALID_ENUM);
        STR(INVALID_VALUE);
        STR(OUT_OF_MEMORY);
        STR(API_UNAVAILABLE);
        STR(VERSION_UNAVAILABLE);
        STR(PLATFORM_ERROR);
        STR(FORMAT_UNAVAILABLE);
#undef STR
        default: return "UNKNOWN_ERROR";
    }
}

constexpr int checkGlfw(int code, std::source_location src = std::source_location::current())
{
    if (code != GLFW_TRUE)
    {
        spdlog::error("GLFW assertion: {} in function {} at {}({})",
            getGlfwErrorString(code), src.function_name(), src.file_name(), src.line()
        );
    }
    return code;
}

} // namespace mimir::validation

namespace mimir
{

// Helper to retrieve engine pointer from handle associated to GLFW window object
MimirInstance *getHandler(GLFWwindow *window)
{
    return reinterpret_cast<MimirInstance*>(glfwGetWindowUserPointer(window));
}

// Translates GLFW mouse movement into Viewer flags for detecting camera movement
void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = getHandler(window);
    auto& ctx = app->window_context;

    auto new_x = static_cast<float>(xpos);
    auto new_y = static_cast<float>(ypos);

    // Fly camera: captured mouse-look. Steer yaw/pitch by the raw cursor delta (clamped pitch to
    // avoid gimbal flip). Signs match the orbit drag below so the feel is consistent.
    if (app->options.camera_control == CameraControl::Fly && ctx.cursor_captured)
    {
        if (ctx.first_mouse) { ctx.mouse_pos = { new_x, new_y }; ctx.first_mouse = false; return; }
        float raw_dx = new_x - ctx.mouse_pos.x;
        float raw_dy = new_y - ctx.mouse_pos.y;
        ctx.mouse_pos = { new_x, new_y };

        float sens = app->options.mouse_sensitivity;
        // Standard FPS mouse-look. Screen y grows downward so a forward mouse push has raw_dy < 0,
        // and rotation.x++ pitches the view DOWN (forward is (0,-sinθ,cosθ)); adding raw_dy thus
        // turns a forward push into a rotation.x decrease = look up.
        app->camera.rotation.y += raw_dx * sens; // yaw:   mouse right   -> look right
        app->camera.rotation.x += raw_dy * sens; // pitch: mouse forward -> look up
        app->camera.rotation.x = std::clamp(app->camera.rotation.x, -89.9f, 89.9f);
        // Roll-free FPS rebuild (keeps the horizon level for any pitch/yaw), not the euler path.
        app->camera.setFlyLook();
        return;
    }

    // Compute displacements from previously registered position
    auto dx = ctx.mouse_pos.x - new_x;
    auto dy = ctx.mouse_pos.y - new_y;

    if (ctx.mouse_buttons.left) // Rotation
    {
        auto speed = app->camera.rotation_speed;
        app->camera.rotate(glm::vec3(dy * speed, -dx * speed, 0.f));
    }
    if (ctx.mouse_buttons.right) // Zoom
    {
        app->camera.translate(glm::vec3(0.f, 0.f, dy * .005f));
    }
    if (ctx.mouse_buttons.middle) // Translation
    {
        app->camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
    }
    // Update last registered mouse position
    ctx.mouse_pos = { .x = new_x, .y = new_y };
}

// Helper to transform button events (pressed, released) into flags (true only if pressed)
bool handleMouseButton(int button, int action, int b)
{
    auto pressed  = (button == b && action == GLFW_PRESS);
    auto released = (button == b && action == GLFW_RELEASE);
    return pressed && !released;
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions
void mouseButtonCallback(GLFWwindow *window, int button, int action,[[maybe_unused]] int mods)
{
    auto app = getHandler(window);
    auto& ctx = app->window_context;

    // Perform action only if GUI does not want mouse input (e.g. not hovering over a menu item)
    if (ImGui::GetIO().WantCaptureMouse) { return; }
    ctx.mouse_buttons = {
        .left   = handleMouseButton(button, action, GLFW_MOUSE_BUTTON_LEFT),
        .right  = handleMouseButton(button, action, GLFW_MOUSE_BUTTON_RIGHT),
        .middle = handleMouseButton(button, action, GLFW_MOUSE_BUTTON_MIDDLE),
    };
}

void framebufferResizeCallback(GLFWwindow *window,[[maybe_unused]] int width,[[maybe_unused]] int height)
{
    auto app = getHandler(window);
    app->window_context.resize_requested = true;
}

void keyCallback(GLFWwindow *window, int key,[[maybe_unused]] int scancode, int action, int mods)
{
    auto app = getHandler(window);
    // Toggle info panel
    if (key == GLFW_KEY_G && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        app->options.show_panel = !app->options.show_panel;
    }
    // Toggle demo window
    if (key == GLFW_KEY_D && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        app->options.show_demo_window = !app->options.show_demo_window;
    }
    // Toggle metrics windows
    if (key == GLFW_KEY_M && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        app->options.show_metrics = !app->options.show_metrics;
    }
    // Trigger exit (useful when window is undecorated)
    if (key == GLFW_KEY_Q && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
        glfwPollEvents();
    }
    // Fly camera: TAB toggles cursor capture (locked for mouse-look vs. free for the ImGui HUD).
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS
        && app->options.camera_control == CameraControl::Fly)
    {
        auto& ctx = app->window_context;
        ctx.cursor_captured = !ctx.cursor_captured;
        ctx.first_mouse = true; // avoid a look jump on the next delta after (re)capturing
        glfwSetInputMode(window, GLFW_CURSOR,
            ctx.cursor_captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
}

void windowCloseCallback(GLFWwindow *window)
{
    spdlog::trace("Triggering window close callback");
    auto engine = getHandler(window);
    engine->signalKernelFinish();
}

inline int bool2glfw(bool flag) { return flag? GLFW_TRUE : GLFW_FALSE; }

GlfwContext GlfwContext::make(WindowOptions options, void *engine)
{
    // Initialize GLFW context and window
    validation::checkGlfw(glfwInit());
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    glfwWindowHint(GLFW_DECORATED, bool2glfw(options.decorate));
    glfwWindowHint(GLFW_VISIBLE, bool2glfw(options.visible));

    auto window = glfwCreateWindow(options.size.x, options.size.y, options.title.c_str(), nullptr, nullptr);
    //glfwSetWindowSize(ctx.window, width, height);

    // Set GLFW action callbacks
    glfwSetWindowUserPointer(window, engine);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);

    return {
        .window           = window,
        .mouse_pos        = { .x = 0.f, .y = 0.f },
        .mouse_buttons    = { .left = false, .right = false, .middle = false },
        .resize_requested = false,
        .cursor_captured  = false,
        .first_mouse      = true,
    };
}

void GlfwContext::clean()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void GlfwContext::exit()
{
    glfwSetWindowShouldClose(window, GL_TRUE);
    glfwPollEvents();
}

bool GlfwContext::shouldClose()
{
    return glfwWindowShouldClose(window);
}

void GlfwContext::processEvents()
{
    glfwPollEvents();
}

void GlfwContext::getFramebufferSize(int& w, int& h)
{
    glfwGetFramebufferSize(window, &w, &h);
}

void GlfwContext::createSurface(VkInstance instance, void *surface)
{
    validation::checkVulkan(
        glfwCreateWindowSurface(instance, window, nullptr, (VkSurfaceKHR*)surface)
    );
}

std::vector<const char*> GlfwContext::getRequiredExtensions()
{
    uint32_t glfw_ext_count = 0;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    return std::vector<const char*>(glfw_exts, glfw_exts + glfw_ext_count);
}

} // namespace mimir