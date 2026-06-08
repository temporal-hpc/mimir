cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(colloids-cuda LANGUAGES C CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_subdirectory(particlesystem)
add_subdirectory(run_delaunay)
