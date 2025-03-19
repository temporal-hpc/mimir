add_executable(benchmark benchmark.cpp nbody_gpu.cu nbody_cpu.cpp)
target_link_libraries(benchmark PRIVATE samples_common powermon)
target_compile_options(benchmark PRIVATE -Wall -Wextra -O3)
set_target_properties(benchmark PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)