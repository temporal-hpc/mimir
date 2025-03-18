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

include(src/base/CMakeLists.txt)
include(src/delaunay/CMakeLists.txt)
include(src/verlet/CMakeLists.txt)

# Include CGAL as an imported target
find_package(CGAL REQUIRED COMPONENTS Core)
target_link_libraries(libparticles mimir curand CGAL::CGAL)
