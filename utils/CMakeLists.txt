add_executable(compile_shaders compile_slang.cpp)
target_link_libraries(compile_shaders PRIVATE slang)

target_compile_features(compile_shaders PUBLIC cxx_std_17)
target_compile_options(compile_shaders PRIVATE -Wall -Wextra -g -O0)