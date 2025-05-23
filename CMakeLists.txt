cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir VERSION 0.1 LANGUAGES CXX CUDA)

option(MIMIR_ENABLE_ASAN "Enable address sanitizer (ASAN)" OFF)
option(MIMIR_BUILD_SLANG "Build slang library from source" OFF)
option(MIMIR_BUILD_SAMPLES "Build sample Mimir programs" ON)

# TODO: Should handle this in multi-platform style
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE
    )
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release"
    )
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(deps) # cmake/deps.cmake

add_subdirectory(lib)
if(MIMIR_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()
