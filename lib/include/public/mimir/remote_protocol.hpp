#pragma once

// Wire protocol for interactive remote rendering (step 2: raw frames over TCP).
// Deliberately dependency-free so a thin native client can include just this header.
// All multi-byte fields are little-endian; for now both ends are assumed same-endian.

#include <cstdint>

namespace mimir::remote
{

// Identifies a mimir remote stream in the Hello message ("MIMR").
constexpr uint32_t PROTOCOL_MAGIC = 0x4D494D52;

// Pixel layout of streamed frames. Matches the engine's offscreen target (B8G8R8A8_UNORM).
enum class PixelFormat : uint32_t { BGRA8 = 0 };

// Control event kinds sent client -> server. Mirror the local mouse interactions.
enum class ControlKind : uint8_t
{
    None         = 0,
    CameraRotate = 1, // a = dx, b = dy (screen-space drag)
    CameraZoom   = 2, // a = dy (wheel/drag)
    CameraPan    = 3, // a = dx, b = dy
    TogglePause  = 4,
    Quit         = 5,
};

#pragma pack(push, 1)

// Sent once by the server immediately after a client connects.
struct Hello
{
    uint32_t magic;  // PROTOCOL_MAGIC
    uint32_t width;
    uint32_t height;
    uint32_t format; // PixelFormat
};

// Precedes each frame's pixel payload (width*height*4 bytes, PixelFormat layout).
struct FrameHeader
{
    uint32_t size; // number of pixel bytes that follow
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
