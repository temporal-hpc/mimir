#pragma once

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>

#include <array> // std::array (vertex buffer)
#include <vector> // std::vector

struct Vertex
{
  glm::vec2 pos;
  glm::vec3 color;
};
