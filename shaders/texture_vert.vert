#version 450

layout(location = 1) in vec2 uniform_texcoords;

layout(location = 0) out vec2 tex_coords;

void main()
{
  tex_coords = uniform_texcoords;
}
