add_executable(run_automata3d CA3D-voxels/main.cu)
find_package(OpenMP)
target_link_libraries(run_automata3d PRIVATE cudaview OpenMP::OpenMP_CXX)

add_executable(run_sync GPU-visual-tool-example/main.cu)
target_link_libraries(run_sync PRIVATE cudaview)

add_executable(image image/run_image.cu)
target_include_directories(image SYSTEM PRIVATE include)
target_link_libraries(image PRIVATE cudaview)

add_executable(mesh3d mesh3d/mesh3d.cu)
target_link_libraries(mesh3d PRIVATE cudaview)

add_executable(points3d points3d/points3d.cu)
target_link_libraries(points3d PRIVATE cudaview)

add_executable(run_structured structured/run_structured.cu)
target_include_directories(run_structured PRIVATE include)
target_link_libraries(run_structured PRIVATE cudaview)

add_executable(run_unstructured unstructured/run_unstructured.cu)
target_include_directories(run_unstructured PRIVATE include)
target_link_libraries(run_unstructured PRIVATE cudaview)

add_executable(run_texture texture_cudaarray/main.cu)
target_include_directories(run_texture PRIVATE include)
target_link_libraries(run_texture PRIVATE cudaview)

# NOTE: Legacy code for shader compilation target
# Made obsolete by slang, but kept here just in case
# ------------------------------------------------------------------------------
# find_program(GLSLC_EXE NAMES glslc HINTS Vulkan::glslc)
# file(GLOB_RECURSE SHADER_SOURCES ${PROJECT_SOURCE_DIR}/shaders/*)
# foreach(HLSL ${SHADER_SOURCES})
#   get_filename_component(FILE_NAME ${HLSL} NAME_WE)
#   set(SPIRV ${PROJECT_BINARY_DIR}/samples/shaders/${FILE_NAME}.spv)
#   add_custom_command(
#     OUTPUT ${SPIRV}
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_BINARY_DIR}/samples/shaders/
#     COMMAND ${GLSLC_EXE} -xhlsl ${HLSL} -o ${SPIRV}
#     DEPENDS ${HLSL})
#   list(APPEND SPIRV_BINARY_FILES ${SPIRV})
# endforeach(HLSL)

#add_custom_target(shaders ALL DEPENDS ${SPIRV_BINARY_FILES})
#add_custom_command(TARGET run_automata3d POST_BUILD
#  COMMAND ${CMAKE_COMMAND} -E copy_directory
#          ${PROJECT_SOURCE_DIR}/shaders
#          ${CMAKE_CURRENT_BINARY_DIR}/shaders
#)

add_custom_target(copy_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${PROJECT_SOURCE_DIR}/shaders
          ${PROJECT_BINARY_DIR}/samples/shaders/
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMENT "Copying test files..."
  VERBATIM
)
