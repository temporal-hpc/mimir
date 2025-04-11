enable_language(C)
find_package(PCL 1.3 REQUIRED COMPONENTS visualization)

add_executable(compare_pcl pointspcl.cu)
target_link_libraries(compare_pcl PRIVATE powermon ${PCL_LIBRARIES})

set_target_properties(compare_pcl PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
