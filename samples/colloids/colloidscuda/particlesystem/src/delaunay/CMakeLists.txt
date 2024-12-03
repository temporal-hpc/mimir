target_sources(libparticles
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_delaunay.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_io.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_cgal.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_cuda.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/device_data.h
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include/particlesystem/particlesystem_delaunay.h
)

# Define headers for this library. PUBLIC headers are used for compiling the
# library, and will be added to consumers' build paths.
target_include_directories(libparticles PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)