#pragma once

#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp> // TODO: Maybe move to validation

#include <array> // std::array (vertex buffer)
#include <vector> // std::vector

struct UniformBufferObject
{
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct Vertex
{
  glm::vec2 pos;
  glm::vec3 color;
};

const std::vector<Vertex> vertices = {
  { { 0.f, -.5f}, {1.f, 0.f, 0.f} },
  { { .5f,  .5f}, {0.f, 1.f, 0.f} },
  { {-.5f,  .5f}, {0.f, 0.f, 1.f} }
};
