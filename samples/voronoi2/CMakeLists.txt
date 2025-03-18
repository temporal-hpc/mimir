add_executable(voronoi test.cu)
target_link_libraries(voronoi samples_common mimir)
set_target_properties(voronoi PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)