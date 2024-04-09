#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector> // std::vector

namespace mimir
{

struct GlfwContext
{
    GLFWwindow *window = nullptr;

    void init(int width, int height, const char* title);
    void clean();
    void exit();
    bool shouldClose();
    void processEvents();
    void getFramebufferSize(int& w, int& h);
    std::vector<const char*> getRequiredExtensions();
};

} // namespace mimir