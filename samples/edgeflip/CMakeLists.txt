find_package(CUDAToolkit REQUIRED)
include(FetchContent)

FetchContent_Declare(imgui_file_dialog
    GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
    GIT_TAG        f73e29fca08163fdcbd1e58fb7b67c7e56f5fa2e # v0.6.7
    GIT_SHALLOW    ON
    FIND_PACKAGE_ARGS
)
# Check if population has already been performed
FetchContent_GetProperties(imgui_file_dialog)
if(NOT imgui_file_dialog_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(imgui_file_dialog)

    add_library(ImGuiFileDialog STATIC
        ${imgui_file_dialog_SOURCE_DIR}/ImGuiFileDialog.cpp
        ${imgui_file_dialog_SOURCE_DIR}/ImGuiFileDialog.h
    )
    target_include_directories(ImGuiFileDialog SYSTEM PUBLIC
        $<BUILD_INTERFACE:${imgui_file_dialog_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ImGuiFileDialog imgui-app)
endif()

add_library(cleap STATIC)
# Compile cleap.cc with nvcc, so that the compiler can find vector_types.h
set_source_files_properties(cleap/src/cleap.cc PROPERTIES LANGUAGE CUDA)
target_sources(cleap PRIVATE cleap/src/cleap.cu cleap/src/cleap.cc)
target_include_directories(cleap PUBLIC cleap/src)
target_link_directories(cleap PRIVATE CUDA::cudart)

add_executable(tiuque)
target_sources(tiuque PRIVATE main.cpp Application.cpp Mesh.cpp)
target_link_libraries(tiuque PRIVATE samples_common cleap ImGuiFileDialog)