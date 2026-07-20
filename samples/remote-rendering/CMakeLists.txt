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

# CUDA architecture for rr-server's device code. "native" auto-detects the configure-time GPU; pin it
# (e.g. -DMIMIR_CUDA_ARCHITECTURES=103 for a B300) to match the library and avoid PTX-JIT on a GPU the
# build dir was not originally configured for. Keep this in sync with the library's build.
set(MIMIR_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architecture(s) for device code: 'native' (auto-detect at configure) or an sm list, e.g. 103")

# --- rr-server: headless GPU renderer/streamer ---------------------------------------------------
# Renders the k-modal point cloud shared with the particles-kmodal-3d sample (kmodal_sim.cu), so
# the remote client sees the same phong / phong-mesh / path-traced view.
if(NOT MIMIR_RR_CLIENT_ONLY)
    set(KMODAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../particles-kmodal-3d")
    add_executable(rr-server rr-server.cu "${KMODAL_DIR}/kmodal_sim.cu")
    target_include_directories(rr-server PRIVATE "${KMODAL_DIR}")
    # The GPU banner + memory pre-flight (NVENC/NVDEC probe, core tables) live in libmimir now
    # (mimir::queryGpuCapabilities / gpuBanner / memoryBudget); NVML is pulled in transitively.
    target_link_libraries(rr-server PRIVATE samples_common spdlog::spdlog)
    set_target_properties(rr-server PROPERTIES
        CUDA_STANDARD      20
        CUDA_ARCHITECTURES "${MIMIR_CUDA_ARCHITECTURES}"
    )
    # Copy the runtime-compiled shaders next to the binary as a DEPENDENCY-TRACKED step, not
    # POST_BUILD: a POST_BUILD copy only fires when the target relinks, so a pull that changes
    # only .slang files left stale copies in build/shaders (which is what the engine compiles).
    file(GLOB_RECURSE RR_SHADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders/*")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders.stamp"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/../../shaders"
            "$<TARGET_FILE_DIR:rr-server>/shaders"
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/shaders.stamp"
        DEPENDS ${RR_SHADER_FILES}
        COMMENT "Copying shaders next to rr-server"
        VERBATIM
    )
    add_custom_target(rr-server-shaders ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/shaders.stamp")
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
