add_library(samples_common INTERFACE)
target_compile_features(samples_common INTERFACE cxx_std_20)
target_include_directories(samples_common SYSTEM INTERFACE include)
target_link_libraries(samples_common INTERFACE mimir)

# samples
add_subdirectory(unstructured)
add_subdirectory(structured)
add_subdirectory(mesh3d)
add_subdirectory(nbody)
add_subdirectory(texture_cudaarray)
add_subdirectory(colloids)
add_subdirectory(potts)
# Utilities
add_subdirectory(powermon)

# Copy slang shaders to output directory
add_custom_target(copy_shaders ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/shaders
        ${PROJECT_BINARY_DIR}/samples/shaders/
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Copying shaders..."
    VERBATIM
)

add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/samples/assets
        ${PROJECT_BINARY_DIR}/samples/assets/
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Copying assets..."
    VERBATIM
)