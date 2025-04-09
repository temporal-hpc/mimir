# Define an executable
add_executable(modelo-delaunay simulation.cpp configvars.h)

set_target_properties(modelo-delaunay PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/..
)

# Define the libraries this project depends upon
target_link_libraries(modelo-delaunay libparticles)
