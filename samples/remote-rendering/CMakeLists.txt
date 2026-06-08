cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(mimir-sample-remote-rendering LANGUAGES CXX CUDA)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MimirSample.cmake")

# --- rr-server: headless GPU renderer/streamer ---------------------------------------------------
add_executable(rr-server rr-server.cu)
target_link_libraries(rr-server PRIVATE samples_common spdlog::spdlog)
set_target_properties(rr-server PROPERTIES
    CUDA_STANDARD      20
    CUDA_ARCHITECTURES native
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
    add_executable(rr-client rr-client.cpp)
    target_compile_features(rr-client PRIVATE cxx_std_20)
    # Use mimir's public include path (for remote_protocol.hpp) without linking the full library.
    target_include_directories(rr-client PRIVATE
        $<TARGET_PROPERTY:mimir,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(rr-client PRIVATE
        PkgConfig::RR_CLIENT OpenSSL::SSL OpenSSL::Crypto glfw OpenGL::GL Threads::Threads)
else()
    message(STATUS "mimir: ngtcp2/ffmpeg/OpenGL not all found, skipping rr-client "
        "(rr-server still builds)")
endif()
