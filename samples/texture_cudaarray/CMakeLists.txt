add_executable(run_texture main.cu)
target_link_libraries(run_texture PRIVATE samples_common)
set_target_properties(run_texture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)