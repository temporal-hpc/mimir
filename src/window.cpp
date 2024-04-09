#include "internal/window.hpp"

#include <imgui.h>
#include <mimir/mimir.hpp>
#include "internal/camera.hpp"

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
    auto dx = app->mouse_pos.x - static_cast<float>(xpos);
    auto dy = app->mouse_pos.y - static_cast<float>(ypos);

    if (app->mouse_buttons.left)
    {
        auto rot = app->camera->rotation_speed;
        app->camera->rotate(glm::vec3(dy * rot, -dx * rot, 0.f));
        app->view_updated = true;
    }
    if (app->mouse_buttons.right)
    {
        app->camera->translate(glm::vec3(0.f, 0.f, dy * .005f));
    }
    if (app->mouse_buttons.middle)
    {
        app->camera->translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
    }
    app->mouse_pos = make_float2(xpos, ypos);
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions
void mouseButtonCallback(GLFWwindow *window, int button, int action,[[maybe_unused]] int mods)
{
    auto app = getHandler(window);
    // Perform action only if GUI does not want to use mouse input
    // (if not hovering over a menu item)
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (action == GLFW_PRESS)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) app->mouse_buttons.middle = true;
            else if (button == GLFW_MOUSE_BUTTON_LEFT) app->mouse_buttons.left = true;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) app->mouse_buttons.right = true;
        }
        else if (action == GLFW_RELEASE)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) app->mouse_buttons.middle = false;
            else if (button == GLFW_MOUSE_BUTTON_LEFT) app->mouse_buttons.left = false;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) app->mouse_buttons.right = false;
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
        app->show_demo_window = !app->show_demo_window;
    }
    // Toggle metrics windows
    if (key == GLFW_KEY_M && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        app->options.show_metrics = !app->options.show_metrics;
    }
}

void windowCloseCallback(GLFWwindow *window)
{
    //printf("Handling window close\n");
    auto engine = getHandler(window);
    engine->running = false;
    engine->signalKernelFinish();
}

void GlfwContext::init(int width, int height, const char* title)
{
    // Initialize GLFW context and window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    //glfwSetWindowSize(window, width, height);

    // Set GLFW action callbacks
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
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

std::vector<const char*> GlfwContext::getRequiredExtensions()
{
    uint32_t glfw_ext_count = 0;
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    return std::vector<const char*>(glfw_exts, glfw_exts + glfw_ext_count);
}

} // namespace mimir