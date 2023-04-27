#include <cudaview/vk_engine.hpp>
#include "internal/camera.hpp"

#include <algorithm>

void VulkanEngine::setBackgroundColor(float4 color)
{
    bg_color = color;
}

// Translates GLFW mouse movement into Viewer flags for detecting camera movement
void VulkanEngine::handleMouseMove(float x, float y)
{
    auto dx = mouse_pos.x - x;
    auto dy = mouse_pos.y - y;

    if (mouse_buttons.left)
    {
        camera->rotate(glm::vec3(dy * camera->rotation_speed, -dx * camera->rotation_speed, 0.f));
        view_updated = true;
    }
    if (mouse_buttons.right)
    {
        camera->translate(glm::vec3(0.f, 0.f, dy * .005f));
    }
    if (mouse_buttons.middle)
    {
        camera->translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
    }
    mouse_pos = float2{x, y};
}

void VulkanEngine::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    app->handleMouseMove(static_cast<float>(xpos), static_cast<float>(ypos));
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions 
void VulkanEngine::handleMouseButton(int button, int action, [[maybe_unused]] int mods)
{
    switch (action)
    {
        case GLFW_PRESS:
        switch (button)
        {
            case GLFW_MOUSE_BUTTON_LEFT:
            mouse_buttons.left = true;
            break;
            case GLFW_MOUSE_BUTTON_MIDDLE:
            mouse_buttons.middle = true;
            break;
            case GLFW_MOUSE_BUTTON_RIGHT:
            mouse_buttons.right = true;
            break;
            default:
            break;
        }
        break;
        case GLFW_RELEASE:
        switch (button)
        {
            case GLFW_MOUSE_BUTTON_LEFT:
            mouse_buttons.left = false;
            break;
            case GLFW_MOUSE_BUTTON_MIDDLE:
            mouse_buttons.middle = false;
            break;
            case GLFW_MOUSE_BUTTON_RIGHT:
            mouse_buttons.right = false;
            break;
            default:
            break;
        }
        break;
        default:
        break;
    }
}

// Translates GLFW mouse scroll into values for detecting camera zoom in/out 
void VulkanEngine::handleScroll([[maybe_unused]] float xoffset, float yoffset)
{
    // depth = std::clamp(depth + yoffset / 10.f, 0.01f, 0.91f);
    // printf("depth= %f, offset= %f\n", depth, yoffset);
}

void VulkanEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    app->handleMouseButton(button, action, mods);
}

void VulkanEngine::framebufferResizeCallback(GLFWwindow *window,[[maybe_unused]] int width,[[maybe_unused]] int height)
{
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    app->should_resize = true;
}

void VulkanEngine::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
    app->handleScroll(static_cast<float>(xoffset), static_cast<float>(yoffset));
}