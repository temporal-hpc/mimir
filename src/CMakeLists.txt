
add_library(cudaview STATIC)

target_sources(cudaview PRIVATE
    camera.cpp
    controls.cpp
    io.cpp
    validation.cpp
    vk_cudadevice.cpp
    vk_device.cpp
    vk_engine.cpp
    vk_framebuffer.cpp
    vk_initializers.cpp
    vk_pipeline.cpp
    vk_properties.cpp
    vk_swapchain.cpp
)
target_include_directories(cudaview SYSTEM PUBLIC ../include)

set_property(TARGET cudaview PROPERTY CUDA_STANDARD 17)
target_compile_features(cudaview PUBLIC cxx_std_17)
target_compile_options(cudaview PRIVATE -Wall -Wextra 
    "$<$<CONFIG:Debug>:-g -O0>"
    "$<$<CONFIG:Release>:-O3>"
)
# This will print the result of the above generator expression, to check if it is correct
#add_custom_command(TARGET cudaview POST_BUILD COMMAND ${CMAKE_COMMAND} -E echo 
#  "-Wall -Wextra $<$<CONFIG:Debug>:-g -O0> $<$<CONFIG:Release>:-O3>")

find_package(Vulkan REQUIRED COMPONENTS glslc)
find_package(CUDAToolkit)
target_link_libraries(cudaview PUBLIC Vulkan::Vulkan CUDA::cudart glfw imgui slang)

install(
    TARGETS cudaview
    EXPORT cudaview # TODO: Check if this is right
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)