#version 450

layout (location = 0) in vec3 in_normal;
layout (location = 1) in vec3 in_color;
layout (location = 2) in float in_alpha;

layout (location = 0) out vec4 frag_color;

vec3 lambert(vec3 normal, vec3 light, vec3 color)
{
  return color * max(dot(normalize(normal), normalize(light)), 0.f);
}

void main()
{
  vec3 light_front = vec3(0.f, 0.f, 1.f);
  vec3 light_up = vec3(0.f, 1.f, 0.f);
  vec3 light_color = in_color;

  float front_bias = .9f;
  vec3 ambient_light = vec3(.1f);
  vec3 color_front = front_bias * lambert(in_normal, light_front, light_color);
  vec3 color_up = (1.f - front_bias) * lambert(in_normal, light_up, light_color);

  frag_color = vec4(ambient_light + color_front + color_up, in_alpha);
}
