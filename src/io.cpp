#include "cudaview/io.hpp"

#include <dlfcn.h> // dladdr

#include <filesystem> // std::filesystem
#include <fstream> // std::ifstream

namespace io
{

// DEPRECATED: Loads a file and returns its data buffer
// Was used for loading compiled shader files, but slang made this obsolete
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

// Reads an .off file and only loads the points and triangles
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

// Setup the shader path so that the library can actually load them
// Hack-ish, but works for now
std::string getDefaultShaderPath()
{
  // If shaders are installed in library path, set working directory there
  Dl_info dl_info;
  dladdr((void*)getDefaultShaderPath, &dl_info);
  auto lib_pathname = dl_info.dli_fname;
  if (lib_pathname != nullptr)
  {
    std::filesystem::path lib_path(lib_pathname);
    return lib_path.parent_path().string();
  }
  else // Use executable path as working dir
  {
    return std::filesystem::read_symlink("/proc/self/exe").remove_filename();
  }
}

} // namespace io
