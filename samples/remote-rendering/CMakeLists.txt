cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-remote-rendering LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

# --- rr-server: headless GPU renderer/streamer ---------------------------------------------------
# Renders the k-modal point cloud shared with the particles-kmodal-3d sample (kmodal_sim.cu), so
# the remote client sees the same phong / phong-mesh / path-traced view.
set(KMODAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../particles-kmodal-3d")
add_executable(rr-server rr-server.cu "${KMODAL_DIR}/kmodal_sim.cu")
target_include_directories(rr-server PRIVATE "${KMODAL_DIR}")
target_link_libraries(rr-server PRIVATE samples_common spdlog::spdlog)
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

# --- rr-client: thin viewer (wire protocol + ngtcp2 + OpenSSL + ffmpeg + GLFW/GL) ----------------
# Built only when its dependencies are present (the same QUIC + H.264 stack the server uses).
find_package(PkgConfig QUIET)
find_package(OpenSSL QUIET)
find_package(OpenGL QUIET)
find_package(Threads QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(RR_CLIENT IMPORTED_TARGET
        libngtcp2 libngtcp2_crypto_ossl libavcodec libavutil libswscale)
endif()

if(RR_CLIENT_FOUND AND OpenSSL_FOUND AND OpenGL_FOUND AND Threads_FOUND)
    include(CheckSymbolExists)
    set(CMAKE_REQUIRED_INCLUDES ${RR_CLIENT_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${RR_CLIENT_LIBRARIES})
    check_symbol_exists(ngtcp2_conn_get_expiry2 "ngtcp2/ngtcp2.h" HAVE_NGTCP2_CONN_GET_EXPIRY2)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)

    add_executable(rr-client rr-client.cpp)
    target_compile_features(rr-client PRIVATE cxx_std_20)
    # Use mimir's public include path (for remote_protocol.hpp) without linking the full library.
    target_include_directories(rr-client PRIVATE
        $<TARGET_PROPERTY:mimir,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(rr-client PRIVATE
        PkgConfig::RR_CLIENT OpenSSL::SSL OpenSSL::Crypto glfw OpenGL::GL Threads::Threads)
    if(HAVE_NGTCP2_CONN_GET_EXPIRY2)
        target_compile_definitions(rr-client PRIVATE HAVE_NGTCP2_CONN_GET_EXPIRY2)
    endif()
else()
    message(STATUS "mimir: ngtcp2/ffmpeg/OpenGL not all found, skipping rr-client "
        "(rr-server still builds)")
endif()
