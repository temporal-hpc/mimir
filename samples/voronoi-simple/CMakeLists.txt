cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-voronoi-simple LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(run_voronoi voronoi.cu)
target_link_libraries(run_voronoi PRIVATE samples_common)
set_target_properties(run_voronoi PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
