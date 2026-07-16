#pragma once

// Wire protocol for interactive remote rendering (step 2: raw frames over TCP).
// Deliberately dependency-free so a thin native client can include just this header.
// All multi-byte fields are little-endian; for now both ends are assumed same-endian.

#include <cctype>  // std::isalnum/std::toupper for benchmark-filename tags
#include <cstdint>
#include <ctime>   // std::strftime for the date stamp
#include <string>  // benchmark-filename helpers

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
    // Ask the encoder for an IDR on the next frame. Sent by a client that lost a frame on an
    // unreliable (datagram) video path and cannot decode further P-frames until a keyframe.
    RequestKeyframe = 7,
    // First-person "turn the gaze" about the eye: a = dx (yaw), b = dy (pitch). Unlike
    // CameraRotate (a trackball that orbits the scene about the world origin), the eye stays
    // put and only the look direction turns -- used to look around from inside the scene.
    CameraLook   = 8,
    // Fly-camera movement (only meaningful when the server runs the Fly camera, see
    // HELLO_CAMERA_FLY): a = strafe (right +), b = forward (+), in camera-local axes. The server
    // moves the eye along its current view basis (looking up + forward climbs). Ignored by a
    // trackball (Orbit) server.
    CameraMove   = 9,
};

// Identifies the client's authentication message ("MIMA"). The client always sends an AuthMsg
// as the first message on the control channel; the server validates the token only when it was
// started with a non-empty one (empty = accept any client).
constexpr uint32_t AUTH_MAGIC = 0x4D494D41;
constexpr unsigned TOKEN_MAX  = 32;

// Sizes of the server-identity strings carried in Hello (NUL-padded, may be empty).
constexpr unsigned USER_MAX = 16;
constexpr unsigned HOST_MAX = 40;
constexpr unsigned GPU_MAX  = 48; // GPU model name, e.g. "NVIDIA A100-SXM4-40GB"

// Flags in Hello::flags describing the session's camera/interaction model.
enum HelloFlags : uint32_t
{
    // The server runs the Fly (first-person) camera: the client should drive it with mouse-look
    // and WASD (sending CameraLook / CameraMove) instead of the trackball orbit/zoom/pan.
    HELLO_CAMERA_FLY = 1u << 0,
};

// Flags on FrameHeader describing the payload that follows.
enum FrameFlags : uint32_t
{
    FRAME_KEYFRAME = 1u << 0, // the access unit is an IDR/keyframe (safe decode start point)
    FRAME_STATS    = 1u << 1, // payload is a Stats struct, not video (server->client telemetry)
    FRAME_HELLO    = 1u << 2, // payload is a new Hello: stream geometry changed (e.g. resize)
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
    // Where the simulation is running, so the client can show it (HUD): the server-side account
    // and hostname. NUL-padded; empty when the server could not determine them.
    char     user[USER_MAX];
    char     host[HOST_MAX];
    uint32_t flags;  // HelloFlags (e.g. HELLO_CAMERA_FLY); 0 = trackball/orbit session
    char     gpu[GPU_MAX]; // GPU model rendering/encoding the stream (HUD); NUL-padded, may be empty
    // LOD reduction mode for this session (static): 0 = native (per-particle), N = an N^3 voxel
    // grid, one representative per occupied cell. The client shows it in the HUD.
    uint32_t lod_cells;
    // Sim/frame coupling (static): 0 = decoupled (the sim runs on its own thread, OFF the frame
    // latency path), N>=1 = lockstep (N sim steps then one frame, sequentially -- so N*compute per
    // step IS part of the frame's latency). The client subtracts that on-path compute from its
    // network~ estimate in lockstep so the residual stays wire+wait, not sim time.
    uint32_t steps_per_frame;
};

// Precedes each payload on the video channel. When flags has FRAME_STATS the payload is a Stats
// struct; otherwise it is a frame (raw pixels or one H.264 access unit) of 'size' bytes.
struct FrameHeader
{
    uint32_t size;       // number of payload bytes that follow
    uint32_t flags;      // FrameFlags
    // Echo of the newest ControlMsg::stamp_ms the server had received when this frame was
    // produced, echoed exactly once (0 = no new stamp). The client subtracts it from its own
    // clock to measure end-to-end latency (input/heartbeat -> decoded frame) with no clock sync.
    uint32_t echo_stamp;
};

// Precedes each fragment of a video frame sent as an unreliable QUIC DATAGRAM (RFC 9221).
// A frame of frame_bytes is split into MTU-sized fragments; the client reassembles by offset
// and considers the frame complete when frame_bytes distinct payload bytes have arrived. Lost
// datagrams are never retransmitted: an overwritten/expired frame is simply dropped and the
// client requests a keyframe (ControlKind::RequestKeyframe) to resume decoding.
struct DatagramFrag
{
    uint32_t frame_id;    // monotonically increasing per encoded frame (gap = frame(s) lost)
    uint32_t offset;      // byte offset of this fragment's payload within the frame
    uint32_t frame_bytes; // total payload size of the whole frame
    uint32_t flags;       // FrameFlags (FRAME_KEYFRAME)
    uint32_t echo_stamp;  // as FrameHeader::echo_stamp (same value on every fragment of a frame)
};

// Periodic server->client stream telemetry (sent as a FRAME_STATS payload).
struct Stats
{
    uint32_t frames;       // frames sent so far this session
    uint32_t fps_milli;    // current frames/sec * 1000
    uint32_t kbps;         // current video bitrate, kilobits/sec
    uint32_t encode_us;    // mean per-frame production latency, microseconds
    // Simulation progress (lifetime, not per session: the sim also advances unwatched).
    uint64_t step;         // simulation steps completed so far
    uint64_t step_limit;   // step count the server stops at (0 = runs endlessly)
    // Appended after step_limit so a newer client reading an older server's shorter Stats keeps
    // the fields above intact (it copies min(payload, sizeof) and leaves this zero).
    uint32_t encode_std_us; // std-dev of per-frame production latency this window, microseconds
    // Appended (same forward-compat rule -- newer fields at the end, zero when absent). The total
    // number of particles the sim advances, and the current throughput (= particle_count * steps/s)
    // in particles/sec, so the client HUD can show scene size and sim throughput.
    uint64_t particle_count;
    uint64_t particles_per_sec;
    // Appended (same forward-compat rule -- zero when absent). Mean wall-clock time of one sim
    // compute() step over the reporting window, microseconds: the "kernel compute time" shown live
    // in the client HUD and the server logs. 0 = older server, or no step measured yet this window.
    uint32_t compute_us;
    // Mean server-side GPU render time per frame this window, microseconds (raster or the full
    // path-trace build+trace). The client HUD shows it as the "render" stage of the pipeline so the
    // local GPU cost is separable from the remote transfer cost (encode + network + decode).
    uint32_t render_us;
};

// Sent by the client as the first message on the control channel, before any ControlMsg.
struct AuthMsg
{
    uint32_t magic;             // AUTH_MAGIC
    char     token[TOKEN_MAX];  // shared secret (NUL-padded); empty when no auth is used
    char     client[HOST_MAX];  // client's hostname, so the server can tag its benchmark CSV
};

// Fixed-size control message sent client -> server.
struct ControlMsg
{
    uint8_t kind;    // ControlKind
    uint8_t pad[3];
    float   a;
    float   b;
    // Client steady-clock milliseconds when the event was created (0 = unstamped). The server
    // echoes the newest stamp it has seen in the next frame's echo_stamp; the client also sends
    // periodic ControlKind::None heartbeats so latency is sampled even without interaction.
    uint32_t stamp_ms;
};

#pragma pack(pop)

// -------------------------------------------------------------------------------------------------
// Benchmark CSV auto-naming (shared by rr-client and rr-server): both derive the same filename
//   <prefix>-<YYYYMMDD>-rr-<role>-c<client>-s<server>-<gpu>.csv
// from the client/server hostnames and the server GPU, so a run's two files pair up by name.
// -------------------------------------------------------------------------------------------------

// First hostname component, alphanumerics only, uppercased (e.g. "patagon.hpc.cl" -> "PATAGON").
inline std::string hostTag(const std::string& host)
{
    std::string out;
    for (char c : host)
    {
        if (c == '.') { break; } // strip the domain
        if (std::isalnum(static_cast<unsigned char>(c)))
        {
            out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        }
        else if (c == '-' && !out.empty() && out.back() != '-') { out += '-'; }
    }
    return out.empty() ? "UNKNOWN" : out;
}

// Compact GPU tag: drop vendor/marketing words, keep the model, uppercased alnum+dash
// ("NVIDIA A100-SXM4-40GB" -> "A100-SXM4-40GB", "NVIDIA GeForce RTX 4090 Laptop GPU" -> "RTX4090").
inline std::string gpuTag(std::string name)
{
    for (auto& c : name) { c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); }
    for (const char *w : {"NVIDIA", "GEFORCE", "LAPTOP", "GPU"})
    {
        for (size_t p; (p = name.find(w)) != std::string::npos; )
        {
            name.erase(p, std::string(w).size());
        }
    }
    std::string out;
    for (char c : name)
    {
        if (std::isalnum(static_cast<unsigned char>(c))) { out += c; }
        else if (c == '-' && !out.empty() && out.back() != '-') { out += '-'; }
    }
    while (!out.empty() && out.back()  == '-') { out.pop_back(); }
    while (!out.empty() && out.front() == '-') { out.erase(out.begin()); }
    return out.empty() ? "GPU" : out;
}

// Local calendar date as YYYYMMDD.
inline std::string dateStamp()
{
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm);
    return buf;
}

// Assembles the full CSV path from a caller-supplied path+prefix and the run's identities.
inline std::string benchmarkCsvPath(const std::string& prefix, const char *role,
    const std::string& client, const std::string& server, const std::string& gpu)
{
    std::string p = prefix;
    if (!p.empty() && p.back() != '/' && p.back() != '-') { p += '-'; }
    p += dateStamp();
    p += "-rr-"; p += role;
    p += "-c" + hostTag(client);
    p += "-s" + hostTag(server);
    p += "-"  + gpuTag(gpu);
    p += ".csv";
    return p;
}

} // namespace mimir::remote
