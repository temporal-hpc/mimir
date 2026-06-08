cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-gpu-visual-tool-example LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(run_sync main.cu)
target_link_libraries(run_sync PRIVATE samples_common)
set_target_properties(run_sync PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
