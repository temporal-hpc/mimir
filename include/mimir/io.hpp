#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace mimir
{
namespace io
{

std::vector<char> readFile(const std::string& filename);

void loadTriangleMesh(const std::string& filename,
    std::vector<float3>& points, std::vector<uint3>& triangles
);

std::string getDefaultShaderPath();

} // namespace io
} // namespace mimir