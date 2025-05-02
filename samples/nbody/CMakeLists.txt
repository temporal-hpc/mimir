find_package(OpenMP)

add_executable(run_nbody nbody.cpp nbody_gpu.cu nbody_cpu.cpp)
target_link_libraries(run_nbody PRIVATE samples_common powermon OpenMP::OpenMP_CXX)
target_compile_options(run_nbody PRIVATE -Wall -Wextra -O3)
set_target_properties(run_nbody PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)