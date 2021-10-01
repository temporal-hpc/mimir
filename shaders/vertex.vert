#version 450

// Vertex attributes, specified per-vertex in the respective buffer
layout(location = 0) in vec2 in_pos;
//layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 frag_color;

//vec2 positions[3] = vec2[](vec2(0.0, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
//vec3 colors[3] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));

void main()
{
  gl_PointSize = 50.f;
  gl_Position = vec4(in_pos, 0.0, 1.0);
  frag_color = vec3(0, 0, 1);
  //frag_color = in_color;
}
