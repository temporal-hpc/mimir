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

struct PrimitiveParams
{
  glm::vec4 color;
  float size;
};

struct SceneParams
{
  glm::ivec3 extent;
  float depth;
};

struct Vertex {
  glm::vec3 pos;
  glm::vec2 uv;
  //glm::vec3 normal;
};
