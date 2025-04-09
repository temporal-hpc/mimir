add_executable(run_voronoi voronoi.cu)
target_link_libraries(run_voronoi PRIVATE samples_common)
set_target_properties(run_voronoi PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)