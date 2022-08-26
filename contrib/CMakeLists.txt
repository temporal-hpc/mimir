add_library(imgui STATIC)
target_sources(imgui PRIVATE
  imgui/imgui.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui SYSTEM PUBLIC imgui)
