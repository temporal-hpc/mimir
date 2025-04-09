add_executable(run_texture main.cu)
target_link_libraries(run_texture PRIVATE samples_common)
set_target_properties(run_texture PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)