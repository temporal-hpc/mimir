#find_package(CUDAToolkit REQUIRED)
#find_package(GLEW REQUIRED) 
find_package(OpenGL)

add_subdirectory(glad-cmake)

add_executable(pointsgl)
target_sources(pointsgl PRIVATE pointsgl.cu camera.cpp)
target_include_directories(pointsgl SYSTEM PUBLIC ../../include/glm)
target_link_libraries(pointsgl PRIVATE powermon glfw glad OpenGL::OpenGL)