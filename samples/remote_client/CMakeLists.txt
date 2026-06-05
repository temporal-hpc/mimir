# Standalone thin client: only needs the wire-protocol header and sockets — no link to mimir,
# CUDA, or Vulkan. Models the eventual native client's dependency footprint.
add_executable(run_remote_client remote_client.cpp)
target_compile_features(run_remote_client PRIVATE cxx_std_20)
target_include_directories(run_remote_client PRIVATE ${PROJECT_SOURCE_DIR}/lib/include/public)
set_target_properties(run_remote_client PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
