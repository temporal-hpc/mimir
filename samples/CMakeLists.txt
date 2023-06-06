add_executable(run_automata3d CA3D-voxels/main.cu)
find_package(OpenMP)
target_link_libraries(run_automata3d PRIVATE cudaview OpenMP::OpenMP_CXX)

add_executable(run_sync GPU-visual-tool-example/main.cu)
target_link_libraries(run_sync PRIVATE cudaview)

add_executable(image image/run_image.cu)
target_include_directories(image SYSTEM PRIVATE include)
target_link_libraries(image PRIVATE cudaview)

add_executable(mesh3d mesh3d/mesh3d.cu)
target_link_libraries(mesh3d PRIVATE cudaview)

add_executable(points3d points3d/points3d.cu)
target_link_libraries(points3d PRIVATE cudaview)

add_executable(run_structured structured/run_structured.cu)
target_link_libraries(run_structured PRIVATE cudaview)

add_executable(run_unstructured unstructured/run_unstructured.cu)
target_link_libraries(run_unstructured PRIVATE cudaview)

add_executable(run_texture texture_cudaarray/main.cu)
target_include_directories(run_texture SYSTEM PRIVATE include)
target_link_libraries(run_texture PRIVATE cudaview)

# Copy slang shaders to output directory
add_custom_target(copy_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${PROJECT_SOURCE_DIR}/shaders
          ${PROJECT_BINARY_DIR}/samples/shaders/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Copying test files..."
  VERBATIM
)
