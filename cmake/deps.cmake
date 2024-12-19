include(FetchContent)

# Vulkan Memory Allocator (VMA)
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        009ecd192c1289c7529bff248a16cfe896254816 # v3.1.0
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

# Slang shader lib
set(SLANG_VERSION 2024.14.6)
FetchContent_Declare(
    slang
    URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/slang-${SLANG_VERSION}-linux-x86_64.tar.gz"
)
FetchContent_MakeAvailable(slang)
# Add imported target containing the precompiled shared library
add_library(slang UNKNOWN IMPORTED)
set_target_properties(slang PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${slang_SOURCE_DIR}/include
    IMPORTED_LOCATION ${slang_SOURCE_DIR}/lib/libslang.so
)

# set(SLANG_ENABLE_GFX        OFF)
# set(SLANG_ENABLE_SLANGD     OFF)
# set(SLANG_ENABLE_SLANGRT    OFF)
# set(SLANG_ENABLE_TESTS      OFF)
# set(SLANG_ENABLE_EXAMPLES   OFF)
# set(SLANG_ENABLE_REPLAYER   OFF)
# set(SLANG_SLANG_LLVM_FLAVOR DISABLE)
# FetchContent_Declare(slang
#     GIT_REPOSITORY https://github.com/shader-slang/slang.git
#     GIT_TAG        e6cf93e3e638cb981a9be392a2f48ea06acd4e3f # v2024.14.6
#     GIT_SHALLOW    ON
#     FIND_PACKAGE_ARGS
# )
# FetchContent_MakeAvailable(slang)

# GLFW windowing lib
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

# GLM shader math lib
option(GLM_ENABLE_CXX_20 ON)
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863 # 1.0.1
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

# spdlog logging lib
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        27cb4c76708608465c413f6d0e6b8d99a4d84302 # v1.14.1
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)

# Download and generate the targets provided by the above contents
FetchContent_MakeAvailable(vma glfw glm spdlog)

# Imgui Graphical interface lib
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        a9f72ab6818c3e55544378aa44c7659de7e5510f # v1.91.2
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(imgui)
add_library(imgui-app STATIC)
target_include_directories(imgui-app SYSTEM PUBLIC
    $<BUILD_INTERFACE:${imgui_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
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
