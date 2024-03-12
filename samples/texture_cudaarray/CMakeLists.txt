add_executable(run_texture main.cu)
target_include_directories(run_texture SYSTEM PRIVATE ../include)
target_link_libraries(run_texture PRIVATE mimir)
set_target_properties(run_texture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)