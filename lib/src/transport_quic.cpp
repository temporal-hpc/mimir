// QUIC transport for interactive remote rendering (step 4).
//
// Preferred transport for direct connections: UDP + TLS + congestion control, with the video
// and control channels mapped to separate QUIC streams. Built on ngtcp2 (the QUIC protocol
// engine) with its OpenSSL crypto binding for the TLS handshake. Only compiled with real
// functionality when MIMIR_HAVE_QUIC is defined (enabled via the MIMIR_ENABLE_QUIC CMake
// option); otherwise listenQuic() reports the missing support and returns nullptr so callers
// transparently fall back to TCP.

#include "mimir/transport.hpp"

#include <spdlog/spdlog.h>

namespace mimir::remote
{

#ifndef MIMIR_HAVE_QUIC

std::unique_ptr<Transport> listenQuic(uint16_t)
{
    spdlog::error("remote(quic): library built without QUIC support; "
        "rebuild with -DMIMIR_ENABLE_QUIC=ON (needs ngtcp2), or use the TCP transport");
    return nullptr;
}

#endif // !MIMIR_HAVE_QUIC

} // namespace mimir::remote
