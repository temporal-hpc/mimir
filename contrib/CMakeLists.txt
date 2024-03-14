add_subdirectory(glfw-3.4)

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