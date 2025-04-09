add_executable(run_structured run_structured.cu)
target_link_libraries(run_structured PRIVATE samples_common)
set_target_properties(run_structured PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
