#version 450

layout (location = 0) in vec3 in_pos;
layout (location = 1) in int in_color;
//layout (location = 2) in float in_alpha;

layout (location = 0) out vec3 out_pos;
layout (location = 1) out vec3 out_color;
layout (location = 2) out float out_alpha;

layout (binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

/*layout (binding = 1) uniform Params
{
  vec3 offset;
} render;*/

void main()
{
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(in_pos /*+ render.offset*/, 1.f);
  out_pos = in_pos; // + render.offset;
  out_color = vec3(in_color);
  out_alpha = 1.f; //in_alpha;
}
