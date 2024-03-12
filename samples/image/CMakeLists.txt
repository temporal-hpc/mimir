add_executable(run_image image.cu)
target_include_directories(run_image SYSTEM PRIVATE ../include)
target_link_libraries(run_image PRIVATE mimir)
set_target_properties(run_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)