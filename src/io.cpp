#include "cudaview/io.hpp"

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

void loadTriangleMesh(const std::string& filename,
  std::vector<float3>& points, std::vector<uint3>& triangles)
{
  constexpr auto stream_max = std::numeric_limits<std::streamsize>::max();
  std::string temp_string;

  std::ifstream stream(filename);
  stream >> temp_string;
  if (temp_string.compare("OFF") != 0)
  {
    return;
  }
  stream.ignore(stream_max, '\n');
  while (stream.peek() == '#')
  {
    stream.ignore(stream_max, '\n');
  }

  size_t point_count, face_count, edge_count;
  stream >> point_count >> face_count >> edge_count;
  points.reserve(point_count);
  triangles.reserve(edge_count);

  for (size_t i = 0; i < point_count; ++i)
  {
    float3 point;
    stream >> point.x >> point.y >> point.z;
    points.push_back(point);
  }
  for (size_t j = 0; j < face_count; ++j)
  {
    uint3 triangle;
    stream >> temp_string >> triangle.x >> triangle.y >> triangle.z;
    triangles.push_back(triangle);
    stream.ignore(stream_max, '\n');
  }
}

} // namespace io
