add_executable(run_benchmark benchmark.cu nbody_cpu.cpp)
target_link_libraries(run_benchmark PRIVATE samples_common powermon)
target_compile_options(run_benchmark PRIVATE -Wall -Wextra -O3)
set_target_properties(run_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)