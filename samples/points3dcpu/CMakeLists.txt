add_executable(run_points_cpu points3dcpu.cpp)
target_link_libraries(run_points_cpu PRIVATE common powermon OpenMP::OpenMP_CXX)
set_target_properties(run_points_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)