#include "meshprogram.hpp"
#include "cuda_utils.hpp"

#include <fstream> // std::ifstream
#include <vector>

void loadMesh2d(const std::string& filename, std::vector<float2>& coords,
  std::vector<uint3>& triangles)
{
  std::ifstream file{filename, std::ios::in};
  if (!file.is_open())
  {
    throw std::runtime_error("failed to open file!");
  }

  int point_count, triangle_count;
  file >> point_count >> triangle_count;

  coords.reserve(point_count);
  triangles.reserve(triangle_count);
  for (int i = 0; i < point_count; ++i)
  {
    float2 point;
    file >> point.x >> point.y;
    coords.push_back(point);
  }
  for (int j = 0; j < triangle_count; ++j)
  {
    uint3 triangle;
    file >> triangle.x >> triangle.y >> triangle.z;
    triangles.push_back(triangle);
  }
}

MeshProgram::MeshProgram()
{}

void MeshProgram::setInitialState()
{
  std::vector<float2> points;
  std::vector<uint3> triangles;
  loadMesh2d("_out/grid2d.dat", points, triangles);

  checkCuda(cudaMemcpy(d_points, points.data(),
    sizeof(float2) * points.size(), cudaMemcpyHostToDevice)
  );
  checkCuda(cudaMemcpy(d_triangles, triangles.data(),
    sizeof(uint3) * triangles.size(), cudaMemcpyHostToDevice)
  );
}

void MeshProgram::cleanup()
{
  checkCuda(cudaFree(d_points));
  checkCuda(cudaFree(d_triangles));
}

void MeshProgram::runTimestep()
{}
