FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        origin/release
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(tinyobjloader)

add_executable(run_mesh mesh3d.cu mesh_obj.cpp)
target_link_libraries(run_mesh PRIVATE samples_common tinyobjloader glm::glm)
set_target_properties(run_mesh PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)