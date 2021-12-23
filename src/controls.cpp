#include "cudaview/vk_engine.hpp"

void VulkanEngine::handleMouseMove(float x, float y)
{
  auto dx = mouse_pos.x - x;
  auto dy = mouse_pos.y - y;

  if (mouse_buttons.left)
  {
    camera.rotate(glm::vec3(dy * camera.rotation_speed, -dx * camera.rotation_speed, 0.f));
    view_updated = true;
  }
  if (mouse_buttons.right)
  {
    camera.translate(glm::vec3(0.f, 0.f, dy * .005f));
  }
  if (mouse_buttons.middle)
  {
    camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
  }
  mouse_pos = glm::vec2(x, y);
}

void VulkanEngine::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->handleMouseMove(static_cast<float>(xpos), static_cast<float>(ypos));
}

void VulkanEngine::handleMouseButton(int button, int action)
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

void VulkanEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->handleMouseButton(button, action);
}

void VulkanEngine::framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
  auto app = reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
  app->should_resize = true;
}
