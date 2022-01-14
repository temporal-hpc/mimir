#pragma once

#include "color/color.hpp"
#include "cudaview/vk_types.hpp"

namespace color
{

void setColor(float *vk_color, color::rgba<float> color);
glm::vec4 getColor(color::rgba<float> color);

} // namespace color
