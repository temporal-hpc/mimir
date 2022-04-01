#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;
// TODO: Add depth uniform (probably on another binding)

layout(location = 0) out vec3 out_uv;

void main()
{
  float depth = 0.f;
  out_uv = vec3(in_uv, depth);
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(in_pos, 1.f);
}
