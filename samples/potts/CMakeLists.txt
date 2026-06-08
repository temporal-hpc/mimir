cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-potts LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

add_executable(potts3)
target_sources(potts3 PRIVATE potts3.cu)
target_link_libraries(potts3 PRIVATE mimir)
set_target_properties(potts3 PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
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

# Copy the data file next to the binary wherever it lands (umbrella or standalone).
add_custom_command(TARGET potts3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_LIST_DIR}/safeprimes_base32.txt"
        "$<TARGET_FILE_DIR:potts3>"
    VERBATIM
)
