add_library(datoviz UNKNOWN IMPORTED)
set_target_properties(datoviz PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/datoviz/include
    IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/datoviz/libdatoviz.so
)
add_executable(compare_dvz points_dvz.cu)
target_link_libraries(compare_dvz PRIVATE powermon datoviz)

set_target_properties(compare_dvz PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
