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

  static VkVertexInputBindingDescription getBindingDescription()
  {
    VkVertexInputBindingDescription binding_desc{};
    binding_desc.binding = 0;
    binding_desc.stride = sizeof(Vertex);
    binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding_desc;
  }

  static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
  {
    std::array<VkVertexInputAttributeDescription, 2> attribute_desc{};
    attribute_desc[0].binding = 0;
    attribute_desc[0].location = 0;
    attribute_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_desc[0].offset = offsetof(Vertex, pos);
    attribute_desc[1].binding = 0;
    attribute_desc[1].location = 1;
    attribute_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_desc[1].offset = offsetof(Vertex, color);
    return attribute_desc;
  }
};
