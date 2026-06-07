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
};

// Listens on the given TCP port and blocks until exactly one client connects and passes the auth
// check, returning a ready transport (or nullptr on failure/auth rejection). A non-empty token is
// required to match the client's AuthMsg; an empty token accepts any client.
std::unique_ptr<Transport> listenTcp(uint16_t port, const std::string& token);

// Listens for a QUIC client on the given UDP port, blocking until the handshake completes and the
// client passes the auth check. Returns a ready transport (or nullptr on failure/auth rejection/
// if built without QUIC support). Token semantics match listenTcp.
std::unique_ptr<Transport> listenQuic(uint16_t port, const std::string& token);

} // namespace mimir::remote
