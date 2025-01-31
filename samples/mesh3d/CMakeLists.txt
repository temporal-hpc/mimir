FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        origin/release
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(tinyobjloader)

add_executable(run_mesh mesh3d.cu)
target_link_libraries(run_mesh PRIVATE samples_common tinyobjloader)
set_target_properties(run_mesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)