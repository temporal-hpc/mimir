set(CPM_DOWNLOAD_VERSION 0.42.0)
if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
        https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# GLFW windowing lib
# Detect X11 availability so GLFW doesn't fail on headless HPC nodes with no X headers.
find_package(X11 QUIET)
if(X11_FOUND)
    set(_glfw_x11 "GLFW_BUILD_X11 ON")
else()
    message(STATUS "mimir: X11 not found — building GLFW without X11 (headless/null platform)")
    set(_glfw_x11 "GLFW_BUILD_X11 OFF")
endif()

CPMAddPackage(URI "gh:glfw/glfw#3.4"
    OPTIONS "GLFW_BUILD_EXAMPLES OFF" "GLFW_BUILD_TESTS OFF" "GLFW_BUILD_DOCS OFF"
            "GLFW_BUILD_WAYLAND OFF" "${_glfw_x11}"
)

# GLM shader math lib
CPMAddPackage(URI "gh:g-truc/glm#1.0.1"
    OPTIONS "GLM_ENABLE_CXX_20 ON"
)

# spdlog logging lib
CPMAddPackage(URI "gh:gabime/spdlog#v1.15.1"
    OPTIONS "GLM_ENABLE_CXX_20 ON"
)

# Slang shader lib
if(MIMIR_BUILD_SLANG)
    message(CHECK_START "Building slang from source")
    set(SLANG_ENABLE_GFX        OFF)
    set(SLANG_ENABLE_SLANGD     OFF)
    set(SLANG_ENABLE_SLANGRT    OFF)
    set(SLANG_ENABLE_TESTS      OFF)
    set(SLANG_ENABLE_EXAMPLES   OFF)
    set(SLANG_ENABLE_REPLAYER   OFF)
    set(SLANG_SLANG_LLVM_FLAVOR DISABLE)
    FetchContent_Declare(slang
        GIT_REPOSITORY https://github.com/shader-slang/slang.git
        GIT_TAG        a09d554721ecf0c6d967d5f55b3416e3284e71f2 # v2025.6.1
        GIT_SHALLOW    ON
        FIND_PACKAGE_ARGS
    )
    FetchContent_MakeAvailable(slang)
    # Add slang install target manually, since FetchContent does not export targets
    install(TARGETS slang EXPORT MimirTargets)
else()
    set(SLANG_VERSION 2025.6.2)
    message(CHECK_START "Using slang release ${SLANG_VERSION}")
    FetchContent_Declare(slang
        URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.tar.gz"
    )
    FetchContent_MakeAvailable(slang)
    # Add imported target containing the precompiled shared library
    add_library(slang UNKNOWN IMPORTED)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${slang_SOURCE_DIR}/include
        IMPORTED_LOCATION ${slang_SOURCE_DIR}/lib/libslang.so
    )
endif()

# Imgui Graphical interface lib
include(FetchContent)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        dbb5eeaadffb6a3ba6a60de1290312e5802dba5a # v1.91.8
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(imgui)
add_library(imgui STATIC)
target_include_directories(imgui SYSTEM PUBLIC
    $<BUILD_INTERFACE:${imgui_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(imgui glfw)
target_sources(imgui PRIVATE
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
