#version 450

layout(binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

layout(binding = 1) uniform UniformDataParams
{
  ivec2 extent;
} params;

layout(location = 0) in vec2 in_pos;
layout(location = 0) out vec3 frag_color;

void main()
{
  gl_PointSize = 10.f;
  vec2 pos = 2.f * (in_pos / params.extent) - 1.f;
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(pos, 0.0, 1.0);
  frag_color = vec3(0, 1, 0);
  //frag_color = in_color;
}
