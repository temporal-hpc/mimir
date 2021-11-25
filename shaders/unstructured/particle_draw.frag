#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main()
{
  //outColor = vec4(fragColor, 1.0);
  vec2 center_coords = 2.f * gl_PointCoord - 1.f;
  float r = dot(center_coords, center_coords);
  if (r > 1.f) discard;
  outColor = vec4(fragColor, 1.f);
}
