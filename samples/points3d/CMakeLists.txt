cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-points3d LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

if(NOT TARGET powermon)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../powermon" "${CMAKE_BINARY_DIR}/powermon_build")
endif()

add_executable(run_points3d points3d.cu)
target_link_libraries(run_points3d PRIVATE samples_common powermon)
set_target_properties(run_points3d PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
