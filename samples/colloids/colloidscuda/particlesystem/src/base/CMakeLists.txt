target_sources(libparticles
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/randgen.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/simparameters.cc
        ${CMAKE_CURRENT_LIST_DIR}/src/kernel_random.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/kernel_util.cu
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include/base/cuda_check.h
        ${CMAKE_CURRENT_LIST_DIR}/include/base/kernel_random.cuh
        ${CMAKE_CURRENT_LIST_DIR}/include/base/kernel_util.h
        ${CMAKE_CURRENT_LIST_DIR}/include/base/math_functions.cuh
        ${CMAKE_CURRENT_LIST_DIR}/include/base/particlesystem.h
        ${CMAKE_CURRENT_LIST_DIR}/include/base/randgen.h
        ${CMAKE_CURRENT_LIST_DIR}/include/base/simparameters.h
)

# Define headers for this library. PUBLIC headers are used for compiling the
# library, and will be added to consumers' build paths.
target_include_directories(libparticles PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
