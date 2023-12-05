find_package(CUDAToolkit)

add_executable(tensorCA)
set_target_properties(tensorCA PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
target_sources(tensorCA PRIVATE
    src/main.cu
    src/TensorCA2D.cu
    src/Debug.cpp
    src/nvmlPower.cu
    src/Rapl.cpp
    src/StatsCollector.cpp
)
set_target_properties(tensorCA PROPERTIES CUDA_ARCHITECTURES "75")
target_compile_definitions(tensorCA PRIVATE
    BSIZE3DX=32
    BSIZE3DY=16
    BSIZE3DZ=1
    DP=NO
    MEASURE_POWER=NO
    R=1
    NREGIONS_H=1
    NREGIONS_V=13
    SMIN=2
    SMAX=3
    BMIN=3
    BMAX=3
)
target_link_libraries(tensorCA PRIVATE CUDA::nvml mimir)