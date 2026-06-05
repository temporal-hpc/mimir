add_executable(run_remote_server remote_server.cu)
target_link_libraries(run_remote_server PRIVATE samples_common)
set_target_properties(run_remote_server PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
