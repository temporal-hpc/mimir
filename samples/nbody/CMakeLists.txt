cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-nbody LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

if(NOT TARGET powermon)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../powermon" "${CMAKE_BINARY_DIR}/powermon_build")
endif()

find_package(OpenMP)

add_executable(benchmark benchmark.cpp nbody_gpu.cu nbody_cpu.cpp)
target_link_libraries(benchmark PRIVATE samples_common powermon OpenMP::OpenMP_CXX)
target_compile_options(benchmark PRIVATE -Wall -Wextra -O3)
set_target_properties(benchmark PROPERTIES
    CXX_STANDARD               20
    CUDA_STANDARD              20
    CUDA_ARCHITECTURES         native
    RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/bin"
)
add_custom_command(TARGET benchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders"
        "$<TARGET_FILE_DIR:benchmark>/shaders"
    COMMENT "Copying shaders next to benchmark"
    VERBATIM
)
