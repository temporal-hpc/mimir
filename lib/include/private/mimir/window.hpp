#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector> // std::vector

namespace mimir
{

struct GlfwContext
{
    GLFWwindow *window;
    struct { float x, y; } mouse_pos;
    struct { bool left, right, middle; } mouse_buttons;
    bool resize_requested;

    void clean();
    void exit();
    bool shouldClose();
    void processEvents();
    void getFramebufferSize(int& w, int& h);
    void createSurface(VkInstance instance, void *surface);

    static GlfwContext make(int width, int height, const char* title, void *engine);
    static std::vector<const char*> getRequiredExtensions();
};

static_assert(std::is_default_constructible_v<GlfwContext>);
static_assert(std::is_nothrow_default_constructible_v<GlfwContext>);
static_assert(std::is_trivially_default_constructible_v<GlfwContext>);

} // namespace mimir