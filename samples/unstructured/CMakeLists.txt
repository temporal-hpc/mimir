cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-unstructured LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(run_unstructured run_unstructured.cu)
target_link_libraries(run_unstructured PUBLIC samples_common)
set_target_properties(run_unstructured PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
