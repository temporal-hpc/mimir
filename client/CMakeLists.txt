# Installable thin remote-rendering client: `mimir-client`.
#
# This is the same program as the samples/remote-rendering `rr-client` (built from the same
# reference source, so there is no duplicated code), packaged here as a standalone tool that gets
# installed with the library. It depends only on the wire protocol — not on any particular server
# workload — so it can view ANY mimir server built with serveRemote(): point your server at a port,
# then run `mimir-client <host> <port>`. Build the server with no client code of your own.
#
# Built only when its dependencies are present (the same QUIC + H.264 stack the server uses);
# otherwise it is skipped and the library still builds.

find_package(PkgConfig QUIET)
find_package(OpenSSL QUIET)
find_package(OpenGL QUIET)
find_package(Threads QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(MIMIR_CLIENT IMPORTED_TARGET
        libngtcp2 libngtcp2_crypto_ossl libavcodec libavutil libswscale)
endif()

if(MIMIR_CLIENT_FOUND AND OpenSSL_FOUND AND OpenGL_FOUND AND Threads_FOUND)
    add_executable(mimir-client ${PROJECT_SOURCE_DIR}/samples/remote-rendering/rr-client.cpp)
    target_compile_features(mimir-client PRIVATE cxx_std_20)
    target_include_directories(mimir-client PRIVATE ${PROJECT_SOURCE_DIR}/lib/include/public)
    target_link_libraries(mimir-client PRIVATE
        PkgConfig::MIMIR_CLIENT OpenSSL::SSL OpenSSL::Crypto glfw OpenGL::GL Threads::Threads)
    set_target_properties(mimir-client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    install(TARGETS mimir-client RUNTIME DESTINATION bin)
    message(STATUS "mimir: building installable mimir-client (remote viewer)")
else()
    message(STATUS "mimir: ngtcp2/ffmpeg/OpenGL not all found, skipping mimir-client")
endif()
