add_library(mimir STATIC)

target_sources(mimir PRIVATE
    src/engine.cpp
    src/camera.cpp
    src/framebuffer.cpp
    src/framelimit.cpp
    src/gui.cpp
    src/image.cpp
    src/interop.cpp
    src/physical_device.cpp
    src/resources.cpp
    src/shader.cpp
    src/swapchain.cpp
    src/validation.cpp
    src/interop_device.cpp
    src/vulkan_device.cpp
    src/vk_pipeline.cpp
    src/vk_properties.cpp
    src/window.cpp
    src/vma_impl.cpp
)
target_sources(mimir PUBLIC
    FILE_SET  library_headers
    TYPE      HEADERS
    BASE_DIRS include
    FILES
        include/mimir/mimir.hpp
        include/mimir/engine/deletion_queue.hpp
        include/mimir/engine/framebuffer.hpp
        include/mimir/engine/image.hpp
        include/mimir/engine/interop_device.hpp
        include/mimir/engine/interop_view.hpp
        include/mimir/engine/performance_monitor.hpp
        include/mimir/engine/physical_device.hpp
        include/mimir/engine/resources.hpp
        include/mimir/engine/swapchain.hpp
        include/mimir/engine/vulkan_device.hpp
        include/mimir/engine/view_types.hpp
)
target_include_directories(mimir PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_property(TARGET mimir PROPERTY CUDA_STANDARD 20)
set_target_properties(mimir PROPERTIES CUDA_ARCHITECTURES native)
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
target_link_libraries(mimir
    PUBLIC
        Vulkan::Vulkan CUDA::cudart glm::glm glfw imgui-app ImGuiFileDialog
    PRIVATE
        GPUOpen::VulkanMemoryAllocator slang fmt::fmt spdlog::spdlog
)

install(
    TARGETS mimir glm glm-header-only glfw VulkanMemoryAllocator slang fmt spdlog imgui-app ImGuiFileDialog
    EXPORT  MimirTargets
    LIBRARY DESTINATION       lib
    ARCHIVE DESTINATION       lib
    RUNTIME DESTINATION       bin
    INCLUDES DESTINATION      include
    FILE_SET                  library_headers
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    VERSION 0.0.1
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT MimirTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirTargets.cmake"
    #NAMESPACE mimir::
)
configure_file("${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfig.cmake"
    COPYONLY
)

set(ConfigPackageLocation lib/cmake/mimir)
install(EXPORT MimirTargets
    FILE mimirTargets.cmake
    #NAMESPACE mimir::
    DESTINATION ${ConfigPackageLocation}
)
install(
    FILES
        "${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    DESTINATION
        ${ConfigPackageLocation}
    #COMPONENT Devel
)