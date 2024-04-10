include(FetchContent)

set(GLFW_BUILD_WAYLAND OFF) # Do not include Wayland support in GLFW
FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw.git
	GIT_TAG        7b6aead9fb88b3623e3b3725ebb42670cbe4c579 # 3.4
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
	glm
	GIT_REPOSITORY https://github.com/g-truc/glm.git # 1.0.1
	GIT_TAG refs/tags/1.0.1
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG        277ae93c41314ba5f4c7444f37c4319cdf07e8cf # v1.90.4
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(imgui)

    # Set custom variables, policies, etc.
    add_library(imgui STATIC)
    target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR})
    target_link_libraries(imgui glfw)
    target_sources(imgui PRIVATE
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    )
endif()

FetchContent_Declare(
	imgui_file_dialog
	GIT_REPOSITORY https://github.com/aiekick/ImGuiFileDialog.git
	GIT_TAG        f73e29fca08163fdcbd1e58fb7b67c7e56f5fa2e # v0.6.7
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
        ${imgui_file_dialog_SOURCE_DIR}
    )
    target_link_libraries(ImGuiFileDialog imgui)
endif()