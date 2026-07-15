cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# Viewer-only mode: build just the mimir-client remote viewer, skipping the library and everything
# server-side. Needs no CUDA toolkit, Vulkan, or NVIDIA hardware, so any Linux machine
# (Intel/AMD graphics included) can build a viewer for a remote mimir server.
option(MIMIR_RR_CLIENT_ONLY "Build only the mimir-client remote viewer (no CUDA/library)" OFF)

if(MIMIR_RR_CLIENT_ONLY)
    project(mimir VERSION 0.1 LANGUAGES CXX)
else()
    project(mimir VERSION 0.1 LANGUAGES CXX CUDA)
endif()

option(MIMIR_ENABLE_ASAN "Enable address sanitizer (ASAN)" OFF)
option(MIMIR_BUILD_SLANG "Build slang library from source" OFF)
option(MIMIR_BUILD_CLIENT "Build the installable mimir-client remote viewer" ON)

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

# In viewer-only mode nothing is fetched or built besides the viewer: its dependencies
# (system GLFW included) are resolved in client/CMakeLists.txt via MimirRemoteClient.cmake.
if(NOT MIMIR_RR_CLIENT_ONLY)
    include(deps) # cmake/deps.cmake
    add_subdirectory(lib)
endif()
if(MIMIR_BUILD_CLIENT OR MIMIR_RR_CLIENT_ONLY)
    add_subdirectory(client)
endif()
