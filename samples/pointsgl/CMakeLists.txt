#find_package(CUDAToolkit REQUIRED)
#find_package(GLEW REQUIRED)
find_package(OpenGL)

add_subdirectory(glad-cmake)

add_executable(compare_ogl)
target_sources(compare_ogl PRIVATE pointsgl.cu)
target_include_directories(compare_ogl SYSTEM PUBLIC ../../include/glm)
target_link_libraries(compare_ogl PRIVATE samples_common powermon glm glfw glad OpenGL::OpenGL)
set_target_properties(compare_ogl PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)

add_custom_target(copy_ogl_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        glsl
        ${PROJECT_BINARY_DIR}/samples/shaders/
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMENT "Copying GLSL shaders..."
    VERBATIM
)