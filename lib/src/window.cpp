#include <mimir/engine/window.hpp>

#include <imgui.h>

#include <mimir/mimir.hpp>
#include <mimir/engine/camera.hpp>
#include "internal/validation.hpp"


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

MimirEngine *getHandler(GLFWwindow *window)
{
    return reinterpret_cast<MimirEngine*>(glfwGetWindowUserPointer(window));
}

// Translates GLFW mouse movement into Viewer flags for detecting camera movement
void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = getHandler(window);
    auto& ctx = app->window_context;

    auto new_x = static_cast<float>(xpos);
    auto new_y = static_cast<float>(ypos);
    auto dx = ctx.mouse_pos.x - new_x;
    auto dy = ctx.mouse_pos.y - new_y;

    if (ctx.mouse_buttons.left)
    {
        auto rot = app->camera.rotation_speed;
        app->camera.rotate(glm::vec3(dy * rot, -dx * rot, 0.f));
        app->view_updated = true;
    }
    if (ctx.mouse_buttons.right)
    {
        app->camera.translate(glm::vec3(0.f, 0.f, dy * .005f));
    }
    if (ctx.mouse_buttons.middle)
    {
        app->camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
    }
    ctx.mouse_pos = { .x = new_x, .y = new_y };
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions
void mouseButtonCallback(GLFWwindow *window, int button, int action,[[maybe_unused]] int mods)
{
    auto app = getHandler(window);
    auto& ctx = app->window_context;
    // Perform action only if GUI does not want to use mouse input
    // (if not hovering over a menu item)
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (action == GLFW_PRESS)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE)     ctx.mouse_buttons.middle = true;
            else if (button == GLFW_MOUSE_BUTTON_LEFT)  ctx.mouse_buttons.left = true;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) ctx.mouse_buttons.right = true;
        }
        else if (action == GLFW_RELEASE)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE)     ctx.mouse_buttons.middle = false;
            else if (button == GLFW_MOUSE_BUTTON_LEFT)  ctx.mouse_buttons.left = false;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) ctx.mouse_buttons.right = false;
        }
    }
}

void framebufferResizeCallback(GLFWwindow *window,[[maybe_unused]] int width,[[maybe_unused]] int height)
{
    auto app = getHandler(window);
    app->should_resize = true;
}

void keyCallback(GLFWwindow *window, int key,[[maybe_unused]] int scancode, int action, int mods)
{
    auto app = getHandler(window);
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
}

void windowCloseCallback(GLFWwindow *window)
{
    spdlog::trace("Executing window close callback");
    auto engine = getHandler(window);
    engine->running = false;
    engine->signalKernelFinish();
}

GlfwContext GlfwContext::make(int width, int height, const char* title, void *engine)
{
    // Initialize GLFW context and window
    validation::checkGlfw(glfwInit());
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    auto window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    //glfwSetWindowSize(ctx.window, width, height);

    // Set GLFW action callbacks
    glfwSetWindowUserPointer(window, engine);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);

    return {
        .window        = window,
        .mouse_pos     = { .x = 0.f, .y = 0.f },
        .mouse_buttons = { .left = false, .right = false, .middle = false },
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