add_library(mimir STATIC)

target_sources(mimir PRIVATE
    engine.cpp
    camera.cpp
    framelimit.cpp
    performance_monitor.cpp
    ui.cpp
    validation.cpp
    interop_device.cpp
    vulkan_device.cpp
    vk_framebuffer.cpp
    vk_initializers.cpp
    vk_pipeline.cpp
    vk_properties.cpp
    vk_swapchain.cpp
)
target_include_directories(mimir SYSTEM PUBLIC ../include)

set_property(TARGET mimir PROPERTY CUDA_STANDARD 20)
set_target_properties(mimir PROPERTIES CUDA_ARCHITECTURES "75")
target_compile_features(mimir PUBLIC cxx_std_20)
target_compile_options(mimir PRIVATE -Wall -Wextra
    "$<$<CONFIG:Debug>:-g>"
    "$<$<CONFIG:Release>:-O3>"
)
# This will print the result of the above generator expression, to check if it is correct
#add_custom_command(TARGET mimir POST_BUILD COMMAND ${CMAKE_COMMAND} -E echo
#  "-Wall -Wextra $<$<CONFIG:Debug>:-g> $<$<CONFIG:Release>:-O3>")

find_package(Vulkan REQUIRED)
find_package(CUDAToolkit)
target_link_libraries(mimir PUBLIC Vulkan::Vulkan CUDA::cudart slang glfw imgui ImGuiFileDialog)

install(
    TARGETS mimir
    EXPORT mimir # TODO: Check if this is right
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)