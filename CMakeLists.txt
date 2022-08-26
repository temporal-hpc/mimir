cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cudaview VERSION 0.1 LANGUAGES CXX CUDA)

add_library(slang SHARED IMPORTED)
target_include_directories(slang INTERFACE contrib/slang/include)
set_property(TARGET slang PROPERTY IMPORTED_LOCATION
  ${PROJECT_SOURCE_DIR}/contrib/slang/lib/libslang.so
)

add_subdirectory(src)
add_subdirectory(contrib)
add_subdirectory(samples)
#add_subdirectory(shaders)
