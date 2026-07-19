cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-CA3D-voxels LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

find_package(OpenMP)

add_executable(run_automata3d main.cu)
# imgui: the sample draws a top-right Performance HUD via setGuiCallback (see main.cu).
target_link_libraries(run_automata3d PRIVATE samples_common OpenMP::OpenMP_CXX imgui)
set_target_properties(run_automata3d PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)
