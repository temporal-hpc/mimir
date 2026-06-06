# QUIC thin client: wire-protocol header + ngtcp2 + OpenSSL + ffmpeg. No mimir/CUDA/Vulkan link.
# Only built when both ngtcp2 and ffmpeg are available (same deps as the server's QUIC + H.264).
find_package(PkgConfig QUIET)
find_package(OpenSSL QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(QUIC_CLIENT IMPORTED_TARGET
        libngtcp2 libngtcp2_crypto_ossl libavcodec libavutil libswscale)
endif()

if(QUIC_CLIENT_FOUND AND OpenSSL_FOUND)
    add_executable(run_remote_quic remote_quic.cpp)
    target_compile_features(run_remote_quic PRIVATE cxx_std_20)
    target_include_directories(run_remote_quic PRIVATE ${PROJECT_SOURCE_DIR}/lib/include/public)
    target_link_libraries(run_remote_quic PRIVATE PkgConfig::QUIC_CLIENT OpenSSL::SSL OpenSSL::Crypto)
    set_target_properties(run_remote_quic PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )
else()
    message(STATUS "mimir: ngtcp2/ffmpeg not found, skipping run_remote_quic")
endif()
