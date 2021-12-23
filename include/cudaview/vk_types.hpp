#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

struct ModelViewProjection
{
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct ColorParams
{
  glm::vec4 point_color;
  glm::vec4 edge_color;
};

struct SceneParams
{
  glm::ivec3 extent;
};
