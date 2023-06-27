cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cudaview VERSION 0.1 LANGUAGES CXX CUDA)

set(default_build_type "Debug")
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

add_library(slang SHARED IMPORTED)
target_include_directories(slang INTERFACE contrib/slang/include)
set_property(TARGET slang PROPERTY IMPORTED_LOCATION
    ${PROJECT_SOURCE_DIR}/contrib/slang/lib/libslang.so
)

add_subdirectory(contrib)
add_subdirectory(src)
add_subdirectory(samples)
add_subdirectory(utils)
#add_subdirectory(shaders)
