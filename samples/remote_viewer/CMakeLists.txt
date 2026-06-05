# Windowed thin client: GLFW + OpenGL + the wire-protocol header. No mimir/CUDA/Vulkan link.
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
add_executable(run_remote_viewer remote_viewer.cpp)
target_compile_features(run_remote_viewer PRIVATE cxx_std_20)
target_include_directories(run_remote_viewer PRIVATE ${PROJECT_SOURCE_DIR}/lib/include/public)
target_link_libraries(run_remote_viewer PRIVATE glfw OpenGL::GL Threads::Threads)
set_target_properties(run_remote_viewer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
