#include "internal/color.hpp"

namespace color
{

void setColor(float *vk_color, color::rgba<float> color)
{
  vk_color[0] = color::get::red(color);
  vk_color[1] = color::get::green(color);
  vk_color[2] = color::get::blue(color);
  vk_color[3] = color::get::alpha(color);
}

glm::vec4 getColor(color::rgba<float> color)
{
  glm::vec4 colorvec;
  colorvec.x = color::get::red(color);
  colorvec.y = color::get::green(color);
  colorvec.z = color::get::blue(color);
  colorvec.w = color::get::alpha(color);
  return colorvec;
}

} // namespace color
