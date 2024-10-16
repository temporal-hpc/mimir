add_executable(run_image image.cu)
target_link_libraries(run_image PRIVATE samples_common)
set_target_properties(run_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)