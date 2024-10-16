add_executable(run_structured run_structured.cu)
target_link_libraries(run_structured PRIVATE samples_common)
set_target_properties(run_structured PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
