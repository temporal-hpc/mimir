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
        src/metrics.cpp
        src/pipeline.cpp
        src/remote.cpp
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
        target_link_libraries(mimir PRIVATE PkgConfig::FFMPEG)
        message(STATUS "mimir: H.264 remote streaming enabled (ffmpeg ${FFMPEG_libavcodec_VERSION})")
    else()
        message(WARNING "mimir: MIMIR_ENABLE_REMOTE set but ffmpeg not found; raw streaming only")
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
configure_file("${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfig.cmake"
    COPYONLY
)

set(ConfigPackageLocation lib/cmake/mimir)
install(EXPORT MimirTargets
    FILE mimirTargets.cmake
    #NAMESPACE mimir::
    DESTINATION ${ConfigPackageLocation}
)
install(
    FILES
        "${PROJECT_SOURCE_DIR}/cmake/mimirConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mimir/mimirConfigVersion.cmake"
    DESTINATION
        ${ConfigPackageLocation}
    #COMPONENT Devel
)