find_package(CUDAToolkit REQUIRED)
include(FetchContent)

FetchContent_Declare(imgui_file_dialog
    GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
    GIT_TAG        f73e29fca08163fdcbd1e58fb7b67c7e56f5fa2e # v0.6.7
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(imgui_file_dialog)
target_link_libraries(ImGuiFileDialog imgui-app)

add_library(cleap STATIC)
# Compile cleap.cc with nvcc, so that the compiler can find vector_types.h
set_source_files_properties(cleap/src/cleap.cc PROPERTIES LANGUAGE CUDA)
target_sources(cleap PRIVATE cleap/src/cleap.cu cleap/src/cleap.cc)
target_include_directories(cleap PUBLIC cleap/src)
target_link_directories(cleap PRIVATE CUDA::cudart)

add_executable(tiuque)
set_target_properties(tiuque PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
target_sources(tiuque PRIVATE main.cpp Application.cpp Mesh.cpp)
target_link_libraries(tiuque PRIVATE samples_common cleap ImGuiFileDialog)