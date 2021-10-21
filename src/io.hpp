#pragma once

#include <string>
#include <vector>

#include "stb_image.h"

namespace io
{

std::vector<char> readFile(const std::string& filename);
stbi_uc *loadImage(const std::string& filename, int& width, int& height, int& channels);

} // namespace io
