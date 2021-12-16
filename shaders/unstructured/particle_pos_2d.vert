#version 450

layout(binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

layout(binding = 1) uniform UniformDataParams
{
  ivec3 extent;
} params;

// Vertex attributes, specified per-vertex in the respective buffer
layout(location = 0) in vec2 in_pos;
//layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

//vec2 positions[3] = vec2[](vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
//vec3 colors[3] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));

void main()
{
  gl_PointSize = 10.f;
  vec3 pos = 2.f * (vec3(in_pos, 1.f) / params.extent) - 1.f;
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(pos, 1.f);
  frag_color = vec3(0, 0, 1);
  //frag_color = in_color;
}
