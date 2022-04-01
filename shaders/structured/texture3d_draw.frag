#version 450

layout(binding = 3) uniform isampler3D tex_sampler;

layout(location = 0) in vec3 tex_uv;

layout(location = 0) out vec4 frag_color;

void main()
{
  // Display texture coordinates as colors
  //frag_color = vec4(tex_uv, 1.f);

  // Display sampled texture
  ivec4 color = texture(tex_sampler, tex_uv);
  frag_color = vec4(vec3(color.r), 1.f);
}
