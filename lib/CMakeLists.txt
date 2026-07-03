find_package(Vulkan      REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(mimir STATIC)
target_sources(mimir
    PRIVATE
        src/mimir.cpp
        src/formats.cpp
        src/engine.cpp
        src/device.cpp
        src/camera.cpp
        src/framebuffer.cpp
        src/framelimit.cpp
        src/gui.cpp
        src/interop.cpp
        src/denoiser_optix.cpp
        src/metrics.cpp
        src/pipeline.cpp
        src/raytracing.cpp
        src/remote.cpp
        src/transport_tcp.cpp
        src/transport_quic.cpp
        src/resources.cpp
        src/shader.cpp
        src/swapchain.cpp
        src/validation.cpp
        src/window.cpp
    PUBLIC
        FILE_SET  public_headers
        TYPE      HEADERS
        BASE_DIRS include/public
        FILES
            include/public/mimir/mimir.hpp
            include/public/mimir/view.hpp
            include/public/mimir/options.hpp
            include/public/mimir/remote_protocol.hpp
)

target_include_directories(mimir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/public>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/private>
)

set_target_properties(mimir PROPERTIES
    COMPILE_WARNING_AS_ERROR ON
    CUDA_ARCHITECTURES       native
    CUDA_STANDARD            20
)
target_compile_features(mimir PRIVATE cxx_std_20)
# Expose the path to ImGui's bundled TTF fonts so gui.cpp can load a crisp DPI-scaled font.
target_compile_definitions(mimir PRIVATE
    "MIMIR_IMGUI_FONTS_DIR=\"${imgui_SOURCE_DIR}/misc/fonts\""
)
target_compile_options(mimir PRIVATE
    -Wall -Wextra -Wformat -Wformat=2 -Wconversion -Wimplicit-fallthrough
    "$<$<BOOL:${MIMIR_ENABLE_ASAN}>:-fsanitize=address>"
    "$<$<CONFIG:Debug>:-g>"
    "$<$<CONFIG:Release>:-O3>"
)

# Optional H.264 encoding for remote streaming, via system ffmpeg/libav* (pkg-config).
# When enabled and found, remote::Codec::H264 becomes available; otherwise serveRemote()
# transparently falls back to raw frames.
option(MIMIR_ENABLE_REMOTE "Enable H.264-encoded remote streaming (requires ffmpeg)" OFF)
if(MIMIR_ENABLE_REMOTE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG IMPORTED_TARGET
        libavcodec libavutil libswscale
    )
    if(FFMPEG_FOUND)
        target_compile_definitions(mimir PRIVATE MIMIR_HAVE_FFMPEG)
        # CUDA::cuda_driver provides cuCtxGetCurrent for the zero-copy NVENC path (sharing
        # mimir's CUDA context with ffmpeg's hardware frames).
        target_link_libraries(mimir PRIVATE PkgConfig::FFMPEG CUDA::cuda_driver)
        message(STATUS "mimir: H.264 remote streaming enabled (ffmpeg ${FFMPEG_libavcodec_VERSION})")
    else()
        message(WARNING "mimir: MIMIR_ENABLE_REMOTE set but ffmpeg not found; raw streaming only")
    endif()
endif()

# Optional QUIC transport for remote streaming, via system ngtcp2 + its OpenSSL crypto binding
# (pkg-config). When enabled and found, remote::TransportKind::Quic becomes available; otherwise
# serveRemote(..., Quic) reports the missing support and the caller can fall back to TCP.
option(MIMIR_ENABLE_QUIC "Enable QUIC transport for remote streaming (requires ngtcp2)" OFF)
if(MIMIR_ENABLE_QUIC)
    find_package(PkgConfig REQUIRED)
    find_package(OpenSSL REQUIRED)
    pkg_check_modules(NGTCP2 IMPORTED_TARGET libngtcp2 libngtcp2_crypto_ossl)
    if(NGTCP2_FOUND)
        target_compile_definitions(mimir PRIVATE MIMIR_HAVE_QUIC)
        target_link_libraries(mimir PRIVATE PkgConfig::NGTCP2 OpenSSL::SSL OpenSSL::Crypto)
        include(CheckSymbolExists)
        set(CMAKE_REQUIRED_INCLUDES ${NGTCP2_INCLUDE_DIRS})
        set(CMAKE_REQUIRED_LIBRARIES ${NGTCP2_LIBRARIES})
        check_symbol_exists(ngtcp2_conn_get_expiry2 "ngtcp2/ngtcp2.h" HAVE_NGTCP2_CONN_GET_EXPIRY2_LIB)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(HAVE_NGTCP2_CONN_GET_EXPIRY2_LIB)
            target_compile_definitions(mimir PRIVATE HAVE_NGTCP2_CONN_GET_EXPIRY2)
        endif()
        message(STATUS "mimir: QUIC transport enabled (ngtcp2 ${NGTCP2_libngtcp2_VERSION})")
    else()
        message(WARNING "mimir: MIMIR_ENABLE_QUIC set but ngtcp2 not found; TCP transport only")
    endif()
endif()

# Optional OptiX AI denoiser (tensor-core accelerated) for path tracing. When enabled and the
# OptiX headers are found, the path-tracing denoiser uses OptiX on NVIDIA GPUs and transparently
# falls back to the built-in Vulkan-compute à-trous filter otherwise. OptiX is header-only (its
# function table loads at runtime via the CUDA driver), so only headers + CUDA::cuda_driver are
# needed. Point OptiX_ROOT / OPTIX_ROOT at the SDK if it is outside the default search paths.
option(MIMIR_ENABLE_OPTIX "Enable the OptiX AI denoiser (tensor cores) for path tracing" OFF)
if(MIMIR_ENABLE_OPTIX)
    find_path(OPTIX_INCLUDE_DIR optix.h
        HINTS
            "${OptiX_ROOT}/include" "$ENV{OptiX_ROOT}/include"
            "${OPTIX_ROOT}/include" "$ENV{OPTIX_ROOT}/include"
            "$ENV{HOME}/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64/include"
    )
    if(OPTIX_INCLUDE_DIR)
        target_compile_definitions(mimir PRIVATE MIMIR_HAVE_OPTIX)
        target_include_directories(mimir SYSTEM PRIVATE ${OPTIX_INCLUDE_DIR})
        target_link_libraries(mimir PRIVATE CUDA::cuda_driver)
        message(STATUS "mimir: OptiX AI denoiser enabled (headers: ${OPTIX_INCLUDE_DIR})")
    else()
        message(WARNING "mimir: MIMIR_ENABLE_OPTIX set but optix.h not found; à-trous denoiser only")
    endif()
endif()

target_link_libraries(mimir
    PUBLIC
        Vulkan::Vulkan CUDA::cudart
    PRIVATE
        slang glm::glm glfw imgui spdlog::spdlog
        "$<$<BOOL:${MIMIR_ENABLE_ASAN}>:-fsanitize=address>"
)

install(
    TARGETS mimir glm glm-header-only glfw imgui spdlog
    EXPORT  MimirTargets
    LIBRARY DESTINATION       lib
    ARCHIVE DESTINATION       lib
    RUNTIME DESTINATION       bin
    INCLUDES DESTINATION      include
    PUBLIC_HEADER DESTINATION include
    FILE_SET                  public_headers
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    VERSION       0.0.1
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT MimirTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirTargets.cmake"
    #NAMESPACE mimir::
)
set(MIMIR_BUILT_WITH_REMOTE ${MIMIR_ENABLE_REMOTE})
set(MIMIR_BUILT_WITH_QUIC   ${MIMIR_ENABLE_QUIC})
set(MIMIR_SLANG_LIBRARY  "${slang_SOURCE_DIR}/lib/libslang.so")
set(MIMIR_SLANG_INCLUDES "${slang_SOURCE_DIR}/include")
configure_file("${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfig.cmake"
    @ONLY
)

set(ConfigPackageLocation lib/cmake/mimir)
install(EXPORT MimirTargets
    FILE mimirTargets.cmake
    #NAMESPACE mimir::
    DESTINATION ${ConfigPackageLocation}
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    DESTINATION
        ${ConfigPackageLocation}
)
# Install slang shared libraries so installed consumers can find them.
install(
    DIRECTORY "${slang_SOURCE_DIR}/lib/"
    DESTINATION lib
    FILES_MATCHING PATTERN "libslang*.so*"
)
install(
    DIRECTORY "${slang_SOURCE_DIR}/include/"
    DESTINATION include/slang
)