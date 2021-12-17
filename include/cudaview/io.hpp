#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace io
{

std::vector<char> readFile(const std::string& filename);

void loadTriangleMesh(const std::string& filename,
  std::vector<float3>& points, std::vector<uint3>& triangles
);

} // namespace io
