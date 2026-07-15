cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-particles-kmodal-3d LANGUAGES CXX CUDA)

# -------------------------------------------------------------------------------------------------
# Shared deps
# -------------------------------------------------------------------------------------------------
if(NOT TARGET powermon)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../powermon" "${CMAKE_BINARY_DIR}/powermon_build")
endif()

find_package(OpenMP)

# -------------------------------------------------------------------------------------------------
# mimir target
# -------------------------------------------------------------------------------------------------
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(benchmark_mimir benchmark_mimir.cpp kmodal_sim.cu)
target_link_libraries(benchmark_mimir PRIVATE samples_common powermon imgui)
target_compile_options(benchmark_mimir PRIVATE -Wall -Wextra -O3)
set_target_properties(benchmark_mimir PROPERTIES
    CXX_STANDARD             20
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
# Dependency-tracked shader copy (not POST_BUILD, which skips when only .slang files changed
# and leaves stale copies in bin/shaders -- the directory the engine actually compiles from).
file(GLOB_RECURSE KMODAL_SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders/*")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders.stamp"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders"
        "$<TARGET_FILE_DIR:benchmark_mimir>/shaders"
    COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/shaders.stamp"
    DEPENDS ${KMODAL_SHADER_FILES}
    COMMENT "Copying shaders next to benchmark_mimir"
    VERBATIM
)
add_custom_target(benchmark-mimir-shaders ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/shaders.stamp")

# Headless acceptance check for LightModel::PathTracing (renders the scene to a PPM).
# EXCLUDE_FROM_ALL so normal builds skip it; build explicitly with: --target pt_check.
add_executable(pt_check EXCLUDE_FROM_ALL pt_check.cpp)
target_link_libraries(pt_check PRIVATE samples_common imgui)
target_compile_options(pt_check PRIVATE -Wall -Wextra -O3)
set_target_properties(pt_check PROPERTIES
    CXX_STANDARD             20
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# -------------------------------------------------------------------------------------------------
# datoviz (built from source via ExternalProject — same setup as samples/ca)
# -------------------------------------------------------------------------------------------------
include(ExternalProject)

set(DATOVIZ_TAG    "v0.3.5"                  CACHE STRING "datoviz git tag to build against")
set(DATOVIZ_PREFIX "${CMAKE_BINARY_DIR}/datoviz" CACHE PATH   "datoviz ExternalProject prefix; override to reuse an existing build")
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
    BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --target libdatoviz --parallel 1
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${DATOVIZ_LIB}"
    USES_TERMINAL_DOWNLOAD TRUE
    USES_TERMINAL_BUILD    TRUE
)

file(MAKE_DIRECTORY "${DATOVIZ_INCLUDE}")

add_library(datoviz SHARED IMPORTED)
set_target_properties(datoviz PROPERTIES
    IMPORTED_LOCATION             "${DATOVIZ_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${DATOVIZ_INCLUDE}"
)
add_dependencies(datoviz datoviz_ext)

# -------------------------------------------------------------------------------------------------
# datoviz target
# -------------------------------------------------------------------------------------------------
add_executable(benchmark_datoviz benchmark_datoviz.cpp kmodal_sim.cu)
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
set_target_properties(benchmark_datoviz PROPERTIES
    BUILD_RPATH   "${DATOVIZ_BUILD};${DATOVIZ_VULKAN_LIBDIR}"
    INSTALL_RPATH "${DATOVIZ_BUILD};${DATOVIZ_VULKAN_LIBDIR}"
)
target_compile_options(benchmark_datoviz PRIVATE -Wall -Wextra -O3)
set_target_properties(benchmark_datoviz PROPERTIES
    CXX_STANDARD             20
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
