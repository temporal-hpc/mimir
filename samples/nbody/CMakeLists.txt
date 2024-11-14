add_executable(run_nbody nbody.cpp bodysystemcuda.cu)
target_include_directories(run_nbody PRIVATE include)
target_link_libraries(run_nbody PRIVATE mimir)
set_target_properties(run_nbody PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)