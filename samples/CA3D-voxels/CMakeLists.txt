find_package(OpenMP)

add_executable(run_automata3d main.cu)
target_link_libraries(run_automata3d PRIVATE samples_common OpenMP::OpenMP_CXX)
set_target_properties(run_automata3d PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)