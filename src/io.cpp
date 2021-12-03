#define STB_IMAGE_IMPLEMENTATION
#include "io.hpp"

#include <fstream> // std::ifstream

namespace io
{

std::vector<char> readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  if (!file.is_open())
  {
    throw std::runtime_error("failed to open file!");
  }

  // Use read position to determine filesize and allocate output buffer
  auto filesize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(filesize);

  file.seekg(0);
  file.read(buffer.data(), filesize);
  file.close();
  return buffer;
}

stbi_uc *loadImage(const std::string& filename, int& width, int& height, int& channels)
{
  return stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
}

} // namespace io
