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

struct ViewUniforms
{
    alignas(16) glm::vec4 color;
    alignas(4)  float size;
    alignas(4)  float depth;
    alignas(4)  int element_count;
};

struct SceneUniforms
{
    alignas(16) glm::vec4 bg_color;
    alignas(16) glm::ivec3 extent;
    alignas(8)  glm::ivec2 resolution;
    alignas(16) glm::vec3 camera_pos;
    alignas(16) glm::vec3 light_pos;
    alignas(16) glm::vec4 light_color;
};

