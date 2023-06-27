find_package(CUDAToolkit REQUIRED)
find_package(GLEW REQUIRED) 
find_package(OpenGL)

add_library(cleap STATIC)
# Compile cleap.cc with nvcc, so that the compiler can find vector_types.h
set_source_files_properties(cleap/src/cleap.cc PROPERTIES LANGUAGE CUDA)
target_sources(cleap PRIVATE cleap/src/cleap.cu cleap/src/cleap.cc)
target_include_directories(cleap PUBLIC cleap/src)
target_link_directories(cleap PRIVATE CUDA::cudart)

add_executable(tiuque main.cpp Application.cpp Mesh.cpp)
target_link_libraries(tiuque PRIVATE
    cudaview cleap OpenGL::OpenGL OpenGL::GLX GLEW::GLEW X11
)