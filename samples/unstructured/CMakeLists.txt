add_executable(run_unstructured run_unstructured.cu)
target_link_libraries(run_unstructured PUBLIC samples_common)
set_target_properties(run_unstructured PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)