add_executable(run_unstructured run_unstructured.cu)
target_link_libraries(run_unstructured PRIVATE mimir)
set_target_properties(run_unstructured PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)