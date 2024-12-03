# Define an executable
add_executable(modelo-verlet
               simulation.cpp
               configvars.h)

# Define the libraries this project depends upon
target_link_libraries(modelo-verlet libparticles)
