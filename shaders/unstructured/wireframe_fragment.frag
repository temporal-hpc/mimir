#version 450

layout(binding = 2) uniform ColorParams
{
  vec4 point_color;
  vec4 edge_color;
} colors;

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main()
{
  outColor = colors.edge_color;
}
