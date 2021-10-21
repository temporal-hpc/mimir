#version 450

layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 tex_coord;
layout(location = 0) out vec4 frag_color;

void main()
{
  frag_color = texture(tex_sampler, tex_coord);
}
