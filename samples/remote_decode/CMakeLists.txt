# H.264 decoding thin client: the wire-protocol header + ffmpeg. No mimir/CUDA/Vulkan link.
# Only built when ffmpeg is available (same dependency as the server's encode path).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG_DECODE IMPORTED_TARGET libavcodec libavutil libswscale)
endif()

if(FFMPEG_DECODE_FOUND)
    add_executable(run_remote_decode remote_decode.cpp)
    target_compile_features(run_remote_decode PRIVATE cxx_std_20)
    target_include_directories(run_remote_decode PRIVATE ${PROJECT_SOURCE_DIR}/lib/include/public)
    target_link_libraries(run_remote_decode PRIVATE PkgConfig::FFMPEG_DECODE)
    set_target_properties(run_remote_decode PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
    )
else()
    message(STATUS "mimir: ffmpeg not found, skipping run_remote_decode")
endif()
