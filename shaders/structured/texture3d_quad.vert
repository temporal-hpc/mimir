#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

layout(binding = 1) uniform UniformDataParams
{
  ivec3 extent;
  float depth;
} scene;

layout(location = 0) out vec3 out_uv;

void main()
{
  out_uv = vec3(in_uv, scene.depth);
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(in_pos, 1.f);
}
