#pragma once

// Wire protocol for interactive remote rendering (step 2: raw frames over TCP).
// Deliberately dependency-free so a thin native client can include just this header.
// All multi-byte fields are little-endian; for now both ends are assumed same-endian.

#include <cstdint>

namespace mimir::remote
{

// Identifies a mimir remote stream in the Hello message ("MIMR").
constexpr uint32_t PROTOCOL_MAGIC = 0x4D494D52;

// Pixel layout of raw streamed frames. Matches the engine's offscreen target (B8G8R8A8_UNORM).
enum class PixelFormat : uint32_t { BGRA8 = 0 };

// How each streamed frame payload is encoded.
//   RawBGRA = width*height*4 uncompressed bytes (PixelFormat layout)
//   H264    = one H.264 access unit (Annex B), decode to recover the frame
enum class Codec : uint32_t { RawBGRA = 0, H264 = 1 };

// Which transport carries the session.
//   Tcp  = single TCP connection (everywhere-works fallback, also the ssh -L tunnel path)
//   Quic = QUIC over UDP (preferred for direct connections: TLS, congestion control, streams)
enum class TransportKind : uint32_t { Tcp = 0, Quic = 1 };

// Control event kinds sent client -> server. Mirror the local mouse interactions.
enum class ControlKind : uint8_t
{
    None         = 0,
    CameraRotate = 1, // a = dx, b = dy (screen-space drag)
    CameraZoom   = 2, // a = dy (wheel/drag)
    CameraPan    = 3, // a = dx, b = dy
    TogglePause  = 4,
    Quit         = 5,
    Resize       = 6, // a = new width, b = new height (server re-renders at this resolution)
};

// Identifies the client's authentication message ("MIMA"). The client always sends an AuthMsg
// as the first message on the control channel; the server validates the token only when it was
// started with a non-empty one (empty = accept any client).
constexpr uint32_t AUTH_MAGIC = 0x4D494D41;
constexpr unsigned TOKEN_MAX  = 32;

// Flags on FrameHeader describing the payload that follows.
enum FrameFlags : uint32_t
{
    FRAME_KEYFRAME = 1u << 0, // the access unit is an IDR/keyframe (safe decode start point)
    FRAME_STATS    = 1u << 1, // payload is a Stats struct, not video (server->client telemetry)
};

#pragma pack(push, 1)

// Sent by the server immediately after a client connects, and again whenever the stream geometry
// changes (e.g. after a Resize). Until the next Hello, frames use this width/height/codec.
struct Hello
{
    uint32_t magic;  // PROTOCOL_MAGIC
    uint32_t width;
    uint32_t height;
    uint32_t format; // PixelFormat (pixel layout once decoded)
    uint32_t codec;  // Codec (how each frame payload is encoded)
};

// Precedes each payload on the video channel. When flags has FRAME_STATS the payload is a Stats
// struct; otherwise it is a frame (raw pixels or one H.264 access unit) of 'size' bytes.
struct FrameHeader
{
    uint32_t size;  // number of payload bytes that follow
    uint32_t flags; // FrameFlags
};

// Periodic server->client stream telemetry (sent as a FRAME_STATS payload).
struct Stats
{
    uint32_t frames;       // frames sent so far this session
    uint32_t fps_milli;    // current frames/sec * 1000
    uint32_t kbps;         // current video bitrate, kilobits/sec
    uint32_t encode_us;    // mean per-frame production latency, microseconds
};

// Sent by the client as the first message on the control channel, before any ControlMsg.
struct AuthMsg
{
    uint32_t magic;             // AUTH_MAGIC
    char     token[TOKEN_MAX];  // shared secret (NUL-padded); empty when no auth is used
};

// Fixed-size control message sent client -> server.
struct ControlMsg
{
    uint8_t kind;    // ControlKind
    uint8_t pad[3];
    float   a;
    float   b;
};

#pragma pack(pop)

} // namespace mimir::remote
