find_package(OpenMP)

add_executable(compare_cpu points3dcpu.cpp)
target_compile_features(compare_cpu PUBLIC cxx_std_20)
target_link_libraries(compare_cpu PRIVATE samples_common powermon OpenMP::OpenMP_CXX)
set_target_properties(compare_cpu PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)