cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(colloids-cuda VERSION 0.1 LANGUAGES C CXX CUDA)

add_subdirectory(particlesystem)
add_subdirectory(run_delaunay)
add_subdirectory(run_verlet)
