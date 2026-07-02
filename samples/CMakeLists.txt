cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-samples LANGUAGES CXX CUDA)

include(cmake/MimirSample.cmake)

# All sample binaries land in bin/ so shaders (copied below) are alongside them.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# powermon must come before samples that link it (nbody).
add_subdirectory(powermon)

# Essential samples
add_subdirectory(headless)
add_subdirectory(remote-rendering)
add_subdirectory(unstructured)
add_subdirectory(structured)
add_subdirectory(mesh3d)
add_subdirectory(image)
add_subdirectory(CA3D-voxels)
# Experiments
add_subdirectory(nbody)
# points3d builds standalone only (like samples/ca): its benchmark_datoviz target fetches
# and builds datoviz from source. Use: ./samples-build-from-zero.sh --sample points3d
# Comparison sample: same nbody simulation rendered with datoviz instead of mimir.
# Off by default because it fetches and builds datoviz from source (slow first build).
option(MIMIR_SAMPLES_BUILD_DATOVIZ "Build the nbody-datoviz comparison sample" OFF)
if(MIMIR_SAMPLES_BUILD_DATOVIZ)
    add_subdirectory(nbody-datoviz)
endif()
# These sample directories were removed but their entries were left behind; disabled so the
# tree configures. Re-enable if the samples are restored.
#add_subdirectory(points3dcpu)
#add_subdirectory(pointsgl)
#add_subdirectory(pointspcl)
# Basic samples
add_subdirectory(GPU-visual-tool-example)
add_subdirectory(edgeflip)
add_subdirectory(voronoi-simple)
add_subdirectory(texture_cudaarray)
# Advanced samples
add_subdirectory(colloids)
add_subdirectory(potts)
add_subdirectory(voronoi2)

# Copy mimir shaders next to the built binaries (the engine loads them at runtime).
add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../shaders"
        "${CMAKE_BINARY_DIR}/bin/shaders/"
    COMMENT "Copying shaders..."
    VERBATIM
)
