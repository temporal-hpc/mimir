# Compile slang shaders at library compile-time
# This is a workaround for renderdoc not getting along with slang for some reason
# -------------------------------------------------------------------------------

function(compile_slang input_path entrypoint)
  get_filename_component(FILE_NAME ${input_path} NAME_WE)
  set(SPIRV ${OUTPUT_DIR}/${FILE_NAME}_${entrypoint}.spv)
  execute_process(
    COMMAND ${SLANGC_EXE} ${input_path} -matrix-layout-row-major -I ${INC_PATH} -entry ${entrypoint} -o ${SPIRV}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction(compile_slang)

# Create output directory for compiled SPIR-V shaders
set(OUTPUT_DIR ${PROJECT_BINARY_DIR}/samples/shaders/precompiled)
message(${OUTPUT_DIR})
add_custom_target(external-shaders-dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
)

# Setup compiler and include paths
find_program(SLANGC_EXE NAMES slangc PATHS ${PROJECT_SOURCE_DIR}/contrib/slang/lib)
set(INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
compile_slang("marker.slang" "vertex2dMain")
compile_slang("marker.slang" "geometryMain")
compile_slang("marker.slang" "fragmentMain")

# This loop cannot be used because some entrypoints and specializations are
# unique to the type of shader that is being compiled
# -------------------------------------------------------------------------------
# file(GLOB SHADER_SOURCES ${PROJECT_SOURCE_DIR}/shaders/*.slang)
# foreach(SRC ${SHADER_SOURCES})
#   get_filename_component(FILE_NAME ${SRC} NAME_WE)
#   set(SPIRV ${PROJECT_BINARY_DIR}/samples/shaders/${FILE_NAME}.spv)
#   compile_slang(${SRC} ${SPIRV})
#   list(APPEND SPIRV_BINARY_FILES ${SPIRV})
# endforeach(SRC)
