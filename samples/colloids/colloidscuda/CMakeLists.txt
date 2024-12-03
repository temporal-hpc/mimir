cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(colloids-cuda VERSION 0.1 LANGUAGES C CXX CUDA)

# Set compilation flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -frounding-math")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(particlesystem)
add_subdirectory(run_delaunay)
add_subdirectory(run_verlet)
