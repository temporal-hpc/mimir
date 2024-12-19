# Define an executable
add_executable(voronoi test.cu)

set_target_properties(voronoi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)

# Define the libraries this project depends upon
target_link_libraries(voronoi mimir)
