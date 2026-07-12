cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# Viewer-only mode: build just the rr-client viewer — no CUDA toolkit, Vulkan, NVIDIA hardware,
# or prebuilt mimir library needed — so any Linux laptop (Intel/AMD graphics included) can build
# the viewer and connect to a remote server:
#   cmake -B build -DMIMIR_RR_CLIENT_ONLY=ON && cmake --build build -j
option(MIMIR_RR_CLIENT_ONLY "Build only the rr-client viewer (no CUDA/mimir)" OFF)

if(MIMIR_RR_CLIENT_ONLY)
    project(mimir-sample-remote-rendering LANGUAGES CXX)
else()
    project(mimir-sample-remote-rendering LANGUAGES CXX CUDA)
    include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")
endif()

# --- rr-server: headless GPU renderer/streamer ---------------------------------------------------
# Renders the k-modal point cloud shared with the particles-kmodal-3d sample (kmodal_sim.cu), so
# the remote client sees the same phong / phong-mesh / path-traced view.
if(NOT MIMIR_RR_CLIENT_ONLY)
    set(KMODAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../particles-kmodal-3d")
    add_executable(rr-server rr-server.cu "${KMODAL_DIR}/kmodal_sim.cu")
    target_include_directories(rr-server PRIVATE "${KMODAL_DIR}")
    # NVML (CUDA::nvml, driver stub) for the startup GPU banner: NVENC/NVDEC presence.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(rr-server PRIVATE samples_common spdlog::spdlog CUDA::nvml)
    set_target_properties(rr-server PROPERTIES
        CUDA_STANDARD      20
        CUDA_ARCHITECTURES native
    )
    add_custom_command(TARGET rr-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders"
            "$<TARGET_FILE_DIR:rr-server>/shaders"
        COMMENT "Copying shaders next to rr-server"
        VERBATIM
    )
endif()

# --- rr-client: thin viewer (wire protocol + ffmpeg + GLFW/GL, QUIC/ngtcp2 optional) -------------
# Shared build logic with the installable mimir-client (top-level client/). In a full build it is
# skipped when its core deps (ffmpeg/GLFW/GL) are missing (rr-server still builds); in viewer-only
# mode those missing deps fail the configure instead. QUIC (ngtcp2 + OpenSSL) is optional either
# way: without it the viewer builds TCP-only.
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/MimirRemoteClient.cmake")
set(_client_required "")
if(MIMIR_RR_CLIENT_ONLY)
    set(_client_required REQUIRED)
endif()
mimir_add_remote_client(rr-client ${_client_required})
