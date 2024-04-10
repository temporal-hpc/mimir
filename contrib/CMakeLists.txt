include(FetchContent)

set(GLFW_BUILD_WAYLAND OFF) # Do not include Wayland support in GLFW
FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG        7b6aead9fb88b3623e3b3725ebb42670cbe4c579 # 3.4
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
	glm
	GIT_REPOSITORY https://github.com/g-truc/glm.git
	GIT_TAG refs/tags/1.0.1
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(glm)

add_library(imgui STATIC)
target_include_directories(imgui SYSTEM PUBLIC imgui-1.90.4)
target_link_libraries(imgui glfw)
target_sources(imgui PRIVATE
    imgui-1.90.4/imgui.cpp
    imgui-1.90.4/imgui_draw.cpp
    imgui-1.90.4/imgui_tables.cpp
    imgui-1.90.4/imgui_widgets.cpp
    imgui-1.90.4/imgui_demo.cpp
    imgui-1.90.4/backends/imgui_impl_glfw.cpp
    imgui-1.90.4/backends/imgui_impl_vulkan.cpp
)

add_subdirectory(imgui_file_dialog)