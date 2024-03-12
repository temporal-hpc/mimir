add_executable(run_automata3d main.cu)
target_link_libraries(run_automata3d PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(run_automata3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)