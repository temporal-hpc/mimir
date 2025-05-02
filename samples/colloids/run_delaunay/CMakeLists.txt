# Define an executable
add_executable(run_colloids simulation.cpp configvars.h)

set_target_properties(run_colloids PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)

# Define the libraries this project depends upon
target_link_libraries(run_colloids libparticles)
