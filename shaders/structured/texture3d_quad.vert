#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

layout(location = 0) out vec2 out_uv;

void main()
{
  out_uv = in_uv;
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(in_pos, 1.f);
}
