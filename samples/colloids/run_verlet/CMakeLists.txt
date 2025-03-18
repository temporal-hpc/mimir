add_executable(modelo-verlet simulation.cpp configvars.h)
target_link_libraries(modelo-verlet libparticles)
set_target_properties(modelo-verlet PROPERTIES
    CUDA_STANDARD            20
    CUDA_ARCHITECTURES       native
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/..
)
