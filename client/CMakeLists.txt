# Installable thin remote-rendering client: `mimir-client`.
#
# This is the same program as the samples/remote-rendering `rr-client` (built from the same
# reference source through cmake/MimirRemoteClient.cmake, so there is no duplicated code),
# packaged here as a standalone tool that gets installed with the library. It depends only on the
# wire protocol — not on any particular server workload — so it can view ANY mimir server built
# with serveRemote(): point your server at a port, then run `mimir-client <host> <port>`. Build
# the server with no client code of your own.
#
# In a normal build it is skipped when its core deps (the H.264 stack + GLFW/GL) are missing, and
# the library still builds. With MIMIR_RR_CLIENT_ONLY=ON this viewer is the whole build, so those
# missing deps fail the configure instead. QUIC (ngtcp2 + OpenSSL) is optional: without it the
# viewer builds TCP-only (so it builds even where libngtcp2_crypto_ossl is unavailable).

include(MimirRemoteClient)

set(_client_required "")
if(MIMIR_RR_CLIENT_ONLY)
    set(_client_required REQUIRED)
endif()
mimir_add_remote_client(mimir-client ${_client_required})

if(TARGET mimir-client)
    set_target_properties(mimir-client PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    install(TARGETS mimir-client RUNTIME DESTINATION bin)
endif()
