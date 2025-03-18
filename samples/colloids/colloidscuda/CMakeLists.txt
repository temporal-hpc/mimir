cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(colloids-cuda VERSION 0.1 LANGUAGES C CXX CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(particlesystem)
add_subdirectory(run_delaunay)
add_subdirectory(run_verlet)
