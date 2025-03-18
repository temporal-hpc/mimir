target_sources(libparticles
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_verlet.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_io.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/cuda_verlet.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/devicedata_verlet.h
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include/particlesystem/particlesystem_verlet.h
)

# Define headers for this library. PUBLIC headers are used for compiling the
# library, and will be added to consumers' build paths.
target_include_directories(libparticles PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
