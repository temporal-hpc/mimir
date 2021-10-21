#version 450

layout(binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 tex_uv;
layout(location = 0) out vec4 frag_color;

void main()
{
  // Display texture coordinates as colors
  //frag_color = vec4(tex_uv, 0.f, 1.f);

  // Display sampled texture
  frag_color = texture(tex_sampler, tex_uv);
}
