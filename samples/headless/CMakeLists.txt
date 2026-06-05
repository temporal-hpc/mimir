add_executable(run_headless headless.cu)
target_link_libraries(run_headless PRIVATE samples_common)
set_target_properties(run_headless PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
