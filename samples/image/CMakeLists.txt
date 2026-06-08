cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-image LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(run_image image.cu)
target_link_libraries(run_image PRIVATE samples_common)
set_target_properties(run_image PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
