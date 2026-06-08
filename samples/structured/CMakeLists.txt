cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-structured LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(run_structured run_structured.cu)
target_link_libraries(run_structured PRIVATE samples_common)
set_target_properties(run_structured PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
