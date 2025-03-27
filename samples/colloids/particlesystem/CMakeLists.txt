project(libparticles VERSION 0.1 LANGUAGES C CXX CUDA)
add_library(libparticles "")

# Request to build target with -std=c++11. As this is a public compile feature,
# anything that links to this target will also build with -std=c++11
target_compile_features(libparticles PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files to be built with -dc,
# as the member functions could be called by other libraries and executables
set_target_properties(libparticles PROPERTIES
    CUDA_STANDARD              20
    CUDA_ARCHITECTURES         native
    CUDA_SEPARABLE_COMPILATION ON
)

target_sources(libparticles
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/randgen.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/simparameters.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/kernel_random.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/kernel_util.cu
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include/cuda_check.h
        ${CMAKE_CURRENT_LIST_DIR}/include/kernel_random.cuh
        ${CMAKE_CURRENT_LIST_DIR}/include/kernel_util.h
        ${CMAKE_CURRENT_LIST_DIR}/include/math_functions.cuh
        ${CMAKE_CURRENT_LIST_DIR}/include/particlesystem.h
        ${CMAKE_CURRENT_LIST_DIR}/include/randgen.h
        ${CMAKE_CURRENT_LIST_DIR}/include/simparameters.h
)

target_sources(libparticles
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_delaunay.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_io.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_cgal.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/particlesystem_cuda.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/device_data.h
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include/particlesystem_delaunay.h
)

# Define headers for this library. PUBLIC headers are used for compiling the
# library, and will be added to consumers' build paths.
target_include_directories(libparticles PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# Define headers for this library. PUBLIC headers are used for compiling the
# library, and will be added to consumers' build paths.
target_include_directories(libparticles PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# Include CGAL as an imported target
find_package(CGAL REQUIRED COMPONENTS Core)
target_link_libraries(libparticles mimir curand CGAL::CGAL)
