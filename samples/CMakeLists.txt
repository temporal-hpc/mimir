find_package(OpenMP)

add_subdirectory(powermon)
add_subdirectory(CA3D-voxels)
add_subdirectory(GPU-visual-tool-example)
add_subdirectory(image)
add_subdirectory(mesh3d)
add_subdirectory(points3d)
add_subdirectory(points3dcpu)
add_subdirectory(structured)
add_subdirectory(unstructured)
add_subdirectory(voronoi-simple)
add_subdirectory(texture_cudaarray)

#add_subdirectory(edgeflip)

# Copy slang shaders to output directory
add_custom_target(copy_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${PROJECT_SOURCE_DIR}/shaders
          ${PROJECT_BINARY_DIR}/samples/shaders/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Copying test files..."
  VERBATIM
)
