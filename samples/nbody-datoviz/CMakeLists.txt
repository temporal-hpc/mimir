cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(nbody-datoviz LANGUAGES CXX CUDA)

include(ExternalProject)

# -------------------------------------------------------------------------------------------------
# datoviz (built from source via ExternalProject)
# -------------------------------------------------------------------------------------------------
# This sample does NOT depend on mimir. It renders the same N-body simulation with datoviz so the
# two libraries can be compared head-to-head.
#
# datoviz's own CMake is not consumable via FetchContent_MakeAvailable/add_subdirectory: it
# hardcodes ${CMAKE_SOURCE_DIR}, has no install() rules, and its bundled Vulkan loader + glslc live
# under fixed in-tree paths. So we build it as a fully separate CMake project (ExternalProject) and
# consume the resulting libdatoviz.so + public headers via an IMPORTED target.
#
# Notes baked in from datoviz v0.3.5:
#   * Library target is `libdatoviz` -> produces `libdatoviz.so` in the build-tree root.
#   * `glslc` (shader compiler) and `libvulkan.so` are checked into the datoviz repo
#     (bin/vulkan/linux, libs/vulkan/linux) — no external Vulkan SDK is required to build.
#   * The `fonts`/`textures` targets (always linked into libdatoviz) read from the `data` git
#     submodule, whose .gitmodules URL is SSH (git@github.com:). GIT_CONFIG rewrites it to HTTPS so
#     the recursive submodule clone works without SSH keys.
#   * DATOVIZ_WITH_MSDF=OFF skips msdfgen (known-flaky first build) and DATOVIZ_WITH_CLI=OFF skips
#     tests/demos. This sample only uses disc markers, so neither is needed.
set(DATOVIZ_TAG "v0.3.5" CACHE STRING "datoviz git tag to build against")

set(DATOVIZ_PREFIX  "${CMAKE_BINARY_DIR}/datoviz")
set(DATOVIZ_SRC     "${DATOVIZ_PREFIX}/src/datoviz_ext")
set(DATOVIZ_BUILD   "${DATOVIZ_PREFIX}/src/datoviz_ext-build")
set(DATOVIZ_LIB     "${DATOVIZ_BUILD}/libdatoviz.so")
set(DATOVIZ_INCLUDE "${DATOVIZ_SRC}/include")
set(DATOVIZ_VULKAN_LIBDIR "${DATOVIZ_SRC}/libs/vulkan/linux")

ExternalProject_Add(datoviz_ext
    GIT_REPOSITORY    https://github.com/datoviz/datoviz.git
    GIT_TAG           ${DATOVIZ_TAG}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
    # Only the submodules we need; rewrite the SSH `data` submodule URL to HTTPS.
    GIT_SUBMODULES    "external/imgui" "data"
    GIT_CONFIG        "url.https://github.com/.insteadOf=git@github.com:"
    PREFIX            "${DATOVIZ_PREFIX}"
    SOURCE_DIR        "${DATOVIZ_SRC}"
    BINARY_DIR        "${DATOVIZ_BUILD}"
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DDATOVIZ_WITH_CLI=OFF
        -DDATOVIZ_WITH_MSDF=OFF
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    # Build only the shared library (skips struct_sizes/cli/etc.).
    # --parallel 1: datoviz's embed_resources.cmake has a write race under parallel make
    # (file(WRITE)+file(APPEND) is not atomic); single-job avoids duplicate shader symbols.
    BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --target libdatoviz --parallel 1
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${DATOVIZ_LIB}"
    USES_TERMINAL_DOWNLOAD TRUE
    USES_TERMINAL_BUILD    TRUE
)

# The include dir does not exist until ExternalProject clones; create it so the IMPORTED target's
# INTERFACE_INCLUDE_DIRECTORIES passes CMake's existence check at configure time.
file(MAKE_DIRECTORY "${DATOVIZ_INCLUDE}")

add_library(datoviz SHARED IMPORTED)
set_target_properties(datoviz PROPERTIES
    IMPORTED_LOCATION             "${DATOVIZ_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${DATOVIZ_INCLUDE}"
)
add_dependencies(datoviz datoviz_ext)

# -------------------------------------------------------------------------------------------------
# powermon (NVML/RAPL power + memory metrics), shared with the mimir nbody sample.
# -------------------------------------------------------------------------------------------------
if(NOT TARGET powermon)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../powermon" "${CMAKE_BINARY_DIR}/powermon_build")
endif()

find_package(OpenMP)

# -------------------------------------------------------------------------------------------------
# Benchmark executable
# -------------------------------------------------------------------------------------------------
add_executable(benchmark_datoviz
    benchmark.cpp
    nbody_gpu.cu
    nbody_cpu.cpp
    pack.cu
)

# Shared header-only helpers from the samples tree (checkCuda). No mimir link.
target_include_directories(benchmark_datoviz PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../include"
    "${DATOVIZ_SRC}/external/imgui"
)

set(DATOVIZ_EXTERNAL_LIB "${DATOVIZ_BUILD}/libexternal_sources.a")

target_link_libraries(benchmark_datoviz PRIVATE
    datoviz
    "${DATOVIZ_EXTERNAL_LIB}"
    powermon
    OpenMP::OpenMP_CXX
)

# Runtime search paths: libdatoviz.so lives in the datoviz build tree, and it pulls in the bundled
# libvulkan.so.1 from the datoviz source tree (linked by soname). Bake both into the RPATH so the
# binary runs in place without LD_LIBRARY_PATH.
set_target_properties(benchmark_datoviz PROPERTIES
    BUILD_RPATH   "${DATOVIZ_BUILD};${DATOVIZ_VULKAN_LIBDIR}"
    INSTALL_RPATH "${DATOVIZ_BUILD};${DATOVIZ_VULKAN_LIBDIR}"
)

target_compile_options(benchmark_datoviz PRIVATE -Wall -Wextra -O3)
set_target_properties(benchmark_datoviz PROPERTIES
    CXX_STANDARD               20
    CUDA_STANDARD              20
    CUDA_ARCHITECTURES         native
    RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/bin"
)
