cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir VERSION 0.1 LANGUAGES CXX CUDA)
include(FetchContent)

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

set(SLANG_ENABLE_GFX        OFF)
set(SLANG_ENABLE_SLANGD     OFF)
set(SLANG_ENABLE_SLANGRT    OFF)
set(SLANG_ENABLE_TESTS      OFF)
set(SLANG_ENABLE_EXAMPLES   OFF)
set(SLANG_SLANG_LLVM_FLAVOR DISABLE)
FetchContent_Declare(slang
    GIT_REPOSITORY https://github.com/shader-slang/slang.git
    GIT_TAG        99584007795a6d6b047698f16db123ba90ce7128 # master
    FIND_PACKAGE_ARGS
)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS    OFF)
set(GLFW_BUILD_DOCS     OFF)
set(GLFW_BUILD_WAYLAND  OFF) # Do not include Wayland support in GLFW
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        7b6aead9fb88b3623e3b3725ebb42670cbe4c579 # 3.4
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

set(GLM_ENABLE_CXX_20 ON)
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863 # 1.0.1
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        7c02e204c92545f869e2f04edaab1f19fe8b19fd # v1.13.0
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

# Download and generate the targets provided by the above contents
FetchContent_MakeAvailable(slang glfw glm) # TODO: Add spdlog

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        277ae93c41314ba5f4c7444f37c4319cdf07e8cf # v1.90.4
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(imgui)

    # An 'imgui' target is already imported by slang as a submodule, so
    add_library(imgui-app STATIC)
    target_include_directories(imgui-app SYSTEM PUBLIC ${imgui_SOURCE_DIR})
    target_link_libraries(imgui-app glfw)
    target_sources(imgui-app PRIVATE
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    )
endif()

FetchContent_Declare(imgui_file_dialog
    GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
    GIT_TAG        f73e29fca08163fdcbd1e58fb7b67c7e56f5fa2e # v0.6.7
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
# Check if population has already been performed
FetchContent_GetProperties(imgui_file_dialog)
if(NOT imgui_file_dialog_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(imgui_file_dialog)

    add_library(ImGuiFileDialog STATIC
        ${imgui_file_dialog_SOURCE_DIR}/ImGuiFileDialog.cpp
        ${imgui_file_dialog_SOURCE_DIR}/ImGuiFileDialog.h
    )
    target_include_directories(ImGuiFileDialog SYSTEM PUBLIC
        ${imgui_file_dialog_SOURCE_DIR}
    )
    target_link_libraries(ImGuiFileDialog imgui-app)
endif()

add_subdirectory(lib)
add_subdirectory(samples)
add_subdirectory(utils)
#add_subdirectory(shaders)
