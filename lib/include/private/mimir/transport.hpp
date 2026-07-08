#pragma once

// Server-side transport abstraction for interactive remote rendering (step 4).
//
// A Transport carries one connected client and exposes two logical channels:
//   - video   (server -> client, reliable + ordered): the Hello message followed by
//             length-prefixed frame payloads (FrameHeader + bytes).
//   - control (client -> server): fixed-size ControlMsg structs.
// TcpTransport puts both channels on a single TCP connection; QuicTransport maps them to
// separate QUIC streams. Because QUIC streams are reliable and ordered, the byte-level framing
// above the transport (Hello, FrameHeader + payload, ControlMsg) is identical either way, so
// serveRemote() never needs to know which transport is active.

#include "mimir/remote_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace mimir::remote
{

// Validates a client's AuthMsg against the server's expected token. An empty expected token
// accepts any client (encryption-without-authentication, as on a tunnel's first hop).
inline bool authOk(const AuthMsg& msg, const std::string& token)
{
    if (msg.magic != AUTH_MAGIC) { return false; }
    if (token.empty()) { return true; }
    return std::strncmp(msg.token, token.c_str(), TOKEN_MAX) == 0;
}

struct Transport
{
    virtual ~Transport() = default;

    // Appends bytes to the server->client video channel. Returns false on send error/disconnect.
    virtual bool sendVideo(const void *data, size_t len) = 0;

    // Appends any fully-received control messages to out. Returns false once the peer has
    // disconnected (no more control will arrive).
    virtual bool pollControl(std::vector<ControlMsg>& out) = 0;

    // True when this transport can deliver video frames unreliably (QUIC DATAGRAM negotiated
    // with the peer). TCP and non-datagram QUIC sessions return false and use sendVideo framing.
    virtual bool unreliableVideoReady() const { return false; }

    // Sends one whole video frame as unreliable datagrams (fragmented to MTU, DatagramFrag
    // framing). Lost fragments are never retransmitted. Returns false when the frame was NOT
    // queued because the previous frame is still stuck behind congestion (the caller should
    // force the next frame to be an IDR so the client can resume decoding past the gap).
    virtual bool sendVideoUnreliable(const void*, size_t, uint32_t /*flags*/, uint32_t /*echo_stamp*/)
    { return false; }
};

// Accepts clients without blocking the caller. The server binds once, then polls from its
// simulation loop: the sim keeps running whether or not anyone is watching, and a client
// connecting simply starts a session on the next poll. One client at a time; further connection
// attempts wait in the socket backlog until the active session ends.
struct Listener
{
    virtual ~Listener() = default;

    // Non-blocking check for a new client. Returns a ready (connected + authenticated)
    // transport, or nullptr when none is pending — call again on the next loop iteration.
    // Unauthorized/failed clients are rejected internally and also yield nullptr.
    // (Once a connection attempt IS in progress, the handshake may block briefly.)
    virtual std::unique_ptr<Transport> poll() = 0;
};

// Binds a TCP listener on the given port. A non-empty token is required to match each client's
// AuthMsg; an empty token accepts any client. Returns nullptr on a bind failure.
std::unique_ptr<Listener> makeTcpListener(uint16_t port, const std::string& token);

// Binds a QUIC (UDP) listener on the given port. Token semantics match makeTcpListener.
// Returns nullptr on a bind failure or if built without QUIC support.
std::unique_ptr<Listener> makeQuicListener(uint16_t port, const std::string& token);

} // namespace mimir::remote
