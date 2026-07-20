cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-mesh3d LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")
include(FetchContent)

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
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
)

# Stage the Slang shaders next to the binary. mimir links statically, so at runtime it resolves its
# shader directory relative to the EXECUTABLE and chdirs there to compile them; without this copy the
# pipelines are created with zero shader stages and the first draw loses the device (ERROR_DEVICE_LOST).
add_custom_command(TARGET run_mesh POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders"
        "$<TARGET_FILE_DIR:run_mesh>/shaders"
    COMMENT "Copying shaders next to run_mesh"
)
