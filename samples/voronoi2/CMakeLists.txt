cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-voronoi2 LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(voronoi test.cu)
target_link_libraries(voronoi PRIVATE samples_common mimir)
set_target_properties(voronoi PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
