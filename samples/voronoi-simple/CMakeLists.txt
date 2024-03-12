add_executable(run_voronoi voronoi.cu)
target_link_libraries(run_voronoi PRIVATE mimir)
set_target_properties(run_voronoi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)