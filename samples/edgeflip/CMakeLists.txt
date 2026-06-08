cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-edgeflip LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

FetchContent_Declare(nfd
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG        86d5f2005fe1c00747348a12070fec493ea2407e # v1.2.1
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nfd)

add_library(cleap STATIC)
# Compile cleap.cc with nvcc so it can find vector_types.h
set_source_files_properties(cleap/src/cleap.cc PROPERTIES LANGUAGE CUDA)
target_sources(cleap PRIVATE cleap/src/cleap.cu cleap/src/cleap.cc)
target_include_directories(cleap PUBLIC cleap/src)
target_link_directories(cleap PRIVATE CUDA::cudart)

add_executable(tiuque)
set_target_properties(tiuque PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
target_sources(tiuque PRIVATE main.cpp Application.cpp Mesh.cpp)
target_link_libraries(tiuque PRIVATE samples_common cleap imgui nfd)
