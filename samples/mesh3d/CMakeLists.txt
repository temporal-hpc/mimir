add_executable(run_mesh mesh3d.cu)
target_link_libraries(run_mesh PRIVATE common)
set_target_properties(run_mesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)