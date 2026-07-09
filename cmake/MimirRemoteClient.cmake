# MimirRemoteClient.cmake — shared build logic for the thin remote-rendering viewer.
#
# The viewer is one program built from samples/remote-rendering/rr-client.cpp; it is packaged
# both as the sample binary `rr-client` and as the installable tool `mimir-client`. It depends
# only on the wire-protocol header (mimir/remote_protocol.hpp, header-only) plus
# ngtcp2 + OpenSSL (QUIC), ffmpeg (H.264 decode), and GLFW/OpenGL (window + present) —
# no CUDA, Vulkan, or the mimir library — so it builds on machines with no NVIDIA stack.
#
#   mimir_add_remote_client(<target>)           # skip with a status message if deps are missing
#   mimir_add_remote_client(<target> REQUIRED)  # fail the configure if deps are missing
#
# Defines <target> when (and only when) all dependencies are found.

function(mimir_add_remote_client target)
    set(_required "")
    if("REQUIRED" IN_LIST ARGN)
        set(_required REQUIRED)
    endif()

    # This file lives in <repo>/cmake/, and the viewer only builds in-tree.
    get_filename_component(_repo_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)

    find_package(PkgConfig ${_required} QUIET)
    find_package(OpenSSL ${_required} QUIET)
    find_package(OpenGL ${_required} QUIET)
    find_package(Threads ${_required} QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(MIMIR_RRC ${_required} IMPORTED_TARGET
            libngtcp2 libngtcp2_crypto_ossl libavcodec libavutil libswscale)
    endif()
    # GLFW: prefer a target that already exists (CPM-built in the top-level tree, or exported
    # through the mimir package in sample builds); otherwise use the system package
    # (Arch: glfw, Debian/Ubuntu: libglfw3-dev).
    if(NOT TARGET glfw)
        find_package(glfw3 3.3 ${_required} QUIET CONFIG)
    endif()

    if(NOT (MIMIR_RRC_FOUND AND OpenSSL_FOUND AND OpenGL_FOUND AND Threads_FOUND AND TARGET glfw))
        message(STATUS "mimir: ngtcp2/ffmpeg/OpenGL/GLFW not all found, skipping ${target}")
        return()
    endif()

    include(CheckSymbolExists)
    set(CMAKE_REQUIRED_INCLUDES ${MIMIR_RRC_INCLUDE_DIRS})
    set(CMAKE_REQUIRED_LIBRARIES ${MIMIR_RRC_LIBRARIES})
    check_symbol_exists(ngtcp2_conn_get_expiry2 "ngtcp2/ngtcp2.h" HAVE_NGTCP2_CONN_GET_EXPIRY2)

    add_executable(${target} "${_repo_root}/samples/remote-rendering/rr-client.cpp")
    target_compile_features(${target} PRIVATE cxx_std_20)
    # mimir's public include path, for remote_protocol.hpp — no library link.
    target_include_directories(${target} PRIVATE "${_repo_root}/lib/include/public")
    target_link_libraries(${target} PRIVATE
        PkgConfig::MIMIR_RRC OpenSSL::SSL OpenSSL::Crypto glfw OpenGL::GL Threads::Threads)
    if(HAVE_NGTCP2_CONN_GET_EXPIRY2)
        target_compile_definitions(${target} PRIVATE HAVE_NGTCP2_CONN_GET_EXPIRY2)
    endif()
    message(STATUS "mimir: building remote viewer ${target}")
endfunction()
