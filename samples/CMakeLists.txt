find_package(OpenMP)

add_library(common INTERFACE)
target_compile_features(common INTERFACE cxx_std_20)
target_include_directories(common SYSTEM INTERFACE include)
target_link_libraries(common INTERFACE mimir)

# Essential samples
add_subdirectory(unstructured)
add_subdirectory(structured)
add_subdirectory(mesh3d)
add_subdirectory(CA3D-voxels)
# # Basic samples
# add_subdirectory(GPU-visual-tool-example)
# add_subdirectory(points3d)
# add_subdirectory(points3dcpu)
# add_subdirectory(edgeflip)
# add_subdirectory(image)
# add_subdirectory(voronoi-simple)
# add_subdirectory(texture_cudaarray)
# # Utilities
# add_subdirectory(powermon)

# Copy slang shaders to output directory
add_custom_target(copy_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${PROJECT_SOURCE_DIR}/shaders
          ${PROJECT_BINARY_DIR}/samples/shaders/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Copying test files..."
  VERBATIM
)
