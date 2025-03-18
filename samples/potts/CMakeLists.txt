add_executable(potts3)
target_sources(potts3 PRIVATE potts3.cu)
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
  ${CMAKE_CURRENT_LIST_DIR}/safeprimes_base32.txt
  ${PROJECT_BINARY_DIR}/samples
  #WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Copying data files..."
  VERBATIM
)