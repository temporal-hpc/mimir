
add_library(cudaview STATIC)

target_sources(cudaview PRIVATE
  camera.cpp
  cudaview.cpp
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
target_include_directories(cudaview SYSTEM PUBLIC
  ../include
  ../contrib/slang/include
)

set_property(TARGET cudaview PROPERTY CUDA_STANDARD 17)
target_compile_features(cudaview PUBLIC cxx_std_17)
target_compile_options(cudaview PRIVATE -Wall -Wextra -g -O0)

find_package(Vulkan REQUIRED COMPONENTS glslc)
find_package(CUDAToolkit)
target_link_libraries(cudaview PUBLIC Vulkan::Vulkan CUDA::cudart glfw imgui slang)
