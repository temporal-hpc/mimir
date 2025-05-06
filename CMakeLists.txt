cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir VERSION 0.1 LANGUAGES CXX CUDA)

option(MIMIR_ENABLE_ASAN "Enable address sanitizer (ASAN)" OFF)
option(MIMIR_BUILD_SLANG "Build slang library from source" OFF)
option(MIMIR_BUILD_SAMPLES "Build sample Mimir programs" ON)

# TODO: Should handle this in multi-platform style
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE
    )
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release"
    )
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(deps) # cmake/deps.cmake

find_package(Vulkan      REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(mimir STATIC)
target_sources(mimir
    PRIVATE
        lib/src/mimir.cpp
        lib/src/formats.cpp
        lib/src/engine.cpp
        lib/src/device.cpp
        lib/src/camera.cpp
        lib/src/framebuffer.cpp
        lib/src/framelimit.cpp
        lib/src/gui.cpp
        lib/src/interop.cpp
        lib/src/metrics.cpp
        lib/src/pipeline.cpp
        lib/src/resources.cpp
        lib/src/shader.cpp
        lib/src/swapchain.cpp
        lib/src/validation.cpp
        lib/src/window.cpp
    PUBLIC
        FILE_SET  public_headers
        TYPE      HEADERS
        BASE_DIRS lib/include/public
        FILES
            lib/include/public/mimir/mimir.hpp
            lib/include/public/mimir/view.hpp
            lib/include/public/mimir/options.hpp
)

target_include_directories(mimir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/public>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/include/private>
)

set_target_properties(mimir PROPERTIES
    COMPILE_WARNING_AS_ERROR ON
    CUDA_ARCHITECTURES       native
    CUDA_STANDARD            20
)
target_compile_features(mimir PRIVATE cxx_std_20)
target_compile_options(mimir PRIVATE
    -Wall -Wextra -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough
    "$<$<BOOL:${MIMIR_ENABLE_ASAN}>:-fsanitize=address>"
    "$<$<CONFIG:Debug>:-g>"
    "$<$<CONFIG:Release>:-O3>"
)

target_link_libraries(mimir
    PUBLIC
        Vulkan::Vulkan CUDA::cudart
    PRIVATE
        slang glm::glm glfw imgui spdlog::spdlog
        "$<$<BOOL:${MIMIR_ENABLE_ASAN}>:-fsanitize=address>"
)

install(
    TARGETS mimir glm glm-header-only glfw imgui spdlog
    EXPORT  MimirTargets
    LIBRARY DESTINATION       lib
    ARCHIVE DESTINATION       lib
    RUNTIME DESTINATION       bin
    INCLUDES DESTINATION      include
    PUBLIC_HEADER DESTINATION include
    FILE_SET                  public_headers
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    VERSION       0.0.1
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT MimirTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirTargets.cmake"
    #NAMESPACE mimir::
)
configure_file("${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfig.cmake"
    COPYONLY
)

set(ConfigPackageLocation lib/cmake/mimir)
install(EXPORT MimirTargets
    FILE mimirTargets.cmake
    #NAMESPACE mimir::
    DESTINATION ${ConfigPackageLocation}
)
install(
    FILES
        "${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    DESTINATION
        ${ConfigPackageLocation}
    #COMPONENT Devel
)

if(MIMIR_BUILD_SAMPLES)
    find_package(OpenMP)
    find_package(CUDAToolkit REQUIRED)

    add_library(samples_common INTERFACE)
    target_compile_features(samples_common INTERFACE cxx_std_20)
    target_include_directories(samples_common SYSTEM INTERFACE samples/include)
    target_link_libraries(samples_common INTERFACE mimir)

    # samples
    add_executable(run_unstructured samples/unstructured/run_unstructured.cu)
    target_link_libraries(run_unstructured PUBLIC samples_common)
    set_target_properties(run_unstructured PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )

    add_executable(run_structured samples/structured/run_structured.cu)
    target_link_libraries(run_structured PRIVATE samples_common)
    set_target_properties(run_structured PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )

    FetchContent_Declare(tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
        GIT_TAG        origin/release
        GIT_SHALLOW    ON
        FIND_PACKAGE_ARGS
    )
    FetchContent_MakeAvailable(tinyobjloader)

    add_executable(run_mesh samples/mesh3d/mesh3d.cu samples/mesh3d/mesh_obj.cpp)
    target_link_libraries(run_mesh PRIVATE samples_common tinyobjloader glm::glm)
    set_target_properties(run_mesh PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )

    add_executable(run_nbody
        samples/nbody/nbody.cpp samples/nbody/nbody_gpu.cu samples/nbody/nbody_cpu.cpp
    )
    target_link_libraries(run_nbody PRIVATE samples_common powermon OpenMP::OpenMP_CXX)
    target_compile_options(run_nbody PRIVATE -Wall -Wextra -O3)
    set_target_properties(run_nbody PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )

    add_executable(run_texture samples/texture_cudaarray/main.cu)
    target_link_libraries(run_texture PRIVATE samples_common)
    set_target_properties(run_texture PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )

    add_library(libparticles STATIC)
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
            samples/colloids/particlesystem/src/randgen.cc
            samples/colloids/particlesystem/src/simparameters.cc
            samples/colloids/particlesystem/src/kernel_random.cu
            samples/colloids/particlesystem/src/kernel_util.cu
            samples/colloids/particlesystem/src/particlesystem_delaunay.cc
            samples/colloids/particlesystem/src/particlesystem_io.cc
            samples/colloids/particlesystem/src/particlesystem_cgal.cc
            samples/colloids/particlesystem/src/particlesystem_cuda.cu
            samples/colloids/particlesystem/src/device_data.h
        PUBLIC
            samples/colloids/particlesystem/include/cuda_check.h
            samples/colloids/particlesystem/include/kernel_random.cuh
            samples/colloids/particlesystem/include/kernel_util.h
            samples/colloids/particlesystem/include/math_functions.cuh
            samples/colloids/particlesystem/include/particlesystem.h
            samples/colloids/particlesystem/include/randgen.h
            samples/colloids/particlesystem/include/simparameters.h
            samples/colloids/particlesystem/include/particlesystem_delaunay.h
    )

    # Define headers for this library. PUBLIC headers are used for compiling the
    # library, and will be added to consumers' build paths.
    target_include_directories(libparticles PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/samples/colloids/particlesystem/include
    )

    # Include CGAL as an imported target
    find_package(CGAL REQUIRED COMPONENTS Core)
    target_link_libraries(libparticles mimir curand CGAL::CGAL)

    add_executable(run_colloids
        samples/colloids/run_delaunay/simulation.cpp
        samples/colloids/run_delaunay/configvars.h
    )
    set_target_properties(run_colloids PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )

    # Define the libraries this project depends upon
    target_link_libraries(run_colloids libparticles)

    # Potts model sample
    add_executable(potts3)
    target_sources(potts3 PRIVATE samples/potts/potts3.cu)
    target_link_libraries(potts3 PRIVATE mimir)
    set_target_properties(potts3 PROPERTIES
        CUDA_STANDARD            20
        CUDA_ARCHITECTURES       native
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )
    target_compile_definitions(potts3 PRIVATE
        Q=9
        L=1024
        SAMPLES=1
        TEMP_MIN=0.71f
        TEMP_MAX=0.73f
        DELTA_TEMP=0.002f
        TRAN=2000
        TMAX=8000
        DELTA_T=50
    )

    add_custom_target(copy_data ALL
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_LIST_DIR}/samples/potts/safeprimes_base32.txt
    ${PROJECT_BINARY_DIR}/samples
    #WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Copying data files..."
    VERBATIM
    )

    # Utilities
    add_library(powermon STATIC)
    target_sources(powermon PRIVATE samples/powermon/nvmlPower.cpp samples/powermon/Rapl.cpp)
    target_include_directories(powermon PUBLIC samples/powermon/include)
    target_link_libraries(powermon PUBLIC CUDA::cudart CUDA::nvml)

    # Copy slang shaders to output directory
    add_custom_target(copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_SOURCE_DIR}/shaders
            ${PROJECT_BINARY_DIR}/samples/shaders/
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Copying shaders..."
        VERBATIM
    )

    add_custom_target(copy_assets ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_SOURCE_DIR}/samples/assets
            ${PROJECT_BINARY_DIR}/samples/assets/
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Copying assets..."
        VERBATIM
    )
endif()
