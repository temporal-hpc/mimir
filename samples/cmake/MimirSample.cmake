# MimirSample.cmake — bootstrap for standalone and umbrella mimir sample builds.
#
# Include this near the top of a sample's CMakeLists.txt, after cmake_minimum_required
# and project(). When built via the umbrella samples/CMakeLists.txt, mimir and
# samples_common already exist — this file is then a no-op.
#
# The library location is resolved in order:
#   1. -Dmimir_DIR=<path> passed on the cmake command line
#   2. MIMIR_DIR environment variable
#   3. Standard system search paths (if mimir was installed system-wide)
#
# MIMIR_DIR must point to the directory that contains mimirConfig.cmake:
#   build tree (no install):  export MIMIR_DIR=/path/to/mimir/build/lib/mimir
#   installed to a prefix:    export MIMIR_DIR=/usr/local/lib/cmake/mimir

if(NOT TARGET mimir)
    if(NOT DEFINED mimir_DIR AND DEFINED ENV{MIMIR_DIR})
        set(mimir_DIR "$ENV{MIMIR_DIR}" CACHE PATH "Directory containing mimirConfig.cmake")
    endif()
    find_package(mimir REQUIRED CONFIG)
endif()

if(NOT TARGET samples_common)
    # cmake/ lives directly inside samples/, so .. resolves to the samples root.
    get_filename_component(_mimir_samples_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    add_library(samples_common INTERFACE)
    target_compile_features(samples_common INTERFACE cxx_std_20)
    target_include_directories(samples_common SYSTEM INTERFACE "${_mimir_samples_root}/include")
    target_link_libraries(samples_common INTERFACE mimir)
endif()
