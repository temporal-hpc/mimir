add_executable(run_points_cpu points3dcpu.cpp)
target_compile_features(run_points_cpu PUBLIC cxx_std_20)
target_link_libraries(run_points_cpu PRIVATE mimir powermon OpenMP::OpenMP_CXX)
set_target_properties(run_points_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)