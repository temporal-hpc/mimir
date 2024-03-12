add_executable(run_sync main.cu)
target_link_libraries(run_sync PRIVATE mimir)
set_target_properties(run_sync PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)