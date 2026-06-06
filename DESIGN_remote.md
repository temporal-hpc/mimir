# Design: Interactive Remote Rendering

Status: **Draft / proposal** (no code yet)
Author: Cristobal
Last updated: 2026-06-04

## 1. Goal

Let a GPU server (e.g. a university cluster node) run the CUDA workload **and** render
frames, while a researcher on a thin native client (e.g. a laptop at home, across the
internet) receives the rendered frames and interacts in real time — moving the camera,
pausing the simulation, toggling views, etc.

The client connects out to the server by address (direct `IP:port`, or `localhost:port`
through an SSH tunnel). The server is headless: it renders with no on-screen window.

### Non-goals (for the first version)
- Browser client (the seams are kept transport-agnostic so this stays possible later).
- Multiple simultaneous clients per server (single-client to start).
- Server-side recording / playback.

## 2. Guiding principle: two seams, everything else untouched

The engine's compute and rendering path is **already independent** of where frames are
displayed and where input comes from. Remote rendering only needs to abstract two couplings
that today are hardwired to the local GLFW window:

1. **Present sink** — `MimirInstance::renderFrame()` (`lib/src/engine.cpp:1144`) acquires a
   swapchain image (`vkAcquireNextImageKHR`, `engine.cpp:1192`) tied to a `VkSurfaceKHR` from
   the GLFW window, draws into `framebuffers.handles[image_idx]`, and presents with
   `vkQueuePresentKHR` (`engine.cpp:1304`).
2. **Control source** — input arrives via GLFW callbacks on `GlfwContext`
   (`lib/include/private/mimir/window.hpp`) that mutate `camera`; the render loop in
   `displayAsync()` (`engine.cpp:160`) calls `window_context.processEvents()` each frame.
   "Pause" is simply not calling the user's compute lambda in the loop.

Everything else stays as-is and is orthogonal to remote rendering:
- CUDA↔Vulkan interop via external memory (`allocLinear`, `engine.cpp:390`) and the timeline
  semaphore barrier (`interop::Barrier`, `engine.cpp:1170-1188`).
- View creation, pipelines, the draw itself (`drawElements`, `engine.cpp:1318`).
- Per-frame uniform/camera upload (`updateUniformBuffers`, `engine.cpp:1415`).

By introducing a `FrameSink` and a `ControlSource` abstraction with **local** (default,
today's behavior) and **remote** implementations, existing samples compile and run unchanged.

## 3. High-level architecture

```
  SERVER (cluster, GPU, headless)                 CLIENT (laptop, native)
  ┌───────────────────────────────┐               ┌──────────────────────────┐
  │ MimirInstance                 │               │ mimir-client             │
  │  CUDA kernels → interop mem   │               │                          │
  │  renderFrame() ── draws into ─┼─► OffscreenSink│  ◄── QUIC video stream   │
  │                               │     (image ring)│      NVDEC/ffmpeg decode │
  │  FrameSink::present(img) ─────┼─► NVENC encode  │      blit to window      │
  │                               │     (H.264/HEVC)│                          │
  │                               │        │        │  capture input (GLFW)    │
  │  ControlSource ◄──────────────┼── QUIC ─┼────────┼─► control stream events  │
  │   applies to Camera / paused  │  control stream  │                          │
  └───────────────────────────────┘               └──────────────────────────┘
```

Threading on the server: the **render thread** draws and `present()`s (enqueues a finished
image); a separate **encoder/sender thread** pulls from the ring, encodes with NVENC, and
writes to the QUIC video stream; a **receiver thread** reads the QUIC control stream and posts
events that the render loop applies before the next frame. The existing
`MAX_FRAMES_IN_FLIGHT = 3` (`engine.hpp:29`) gives the slack for render and encode to overlap.

## 4. Server-side design

### 4.1 `FrameSink` — present backend abstraction

Wrap the two swapchain calls `renderFrame()` makes today:

```cpp
struct FrameSink {
    virtual ~FrameSink() = default;
    // Get the next target image to render into; signals imgReady when usable.
    virtual uint32_t acquire(VkSemaphore imgReady) = 0;
    virtual VkImage  image(uint32_t idx) = 0;
    virtual VkExtent2D extent() const = 0;
    virtual VkFormat   format() const = 0;
    // "Present": local = vkQueuePresentKHR; remote = enqueue for encode+send.
    virtual void present(uint32_t idx, VkSemaphore renderDone) = 0;
};
```

- **`WindowSink`** — wraps the current `Swapchain` (`swapchain.hpp`) + `vkQueuePresentKHR`.
  Exactly today's behavior. Default for `RenderMode::Local`.
- **`OffscreenSink`** — owns a ring of N offscreen color images (a "virtual swapchain"),
  created with `createImage` (`resources.hpp:31`) using
  `VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT` (or sampled, for the
  encoder's input view). `acquire()` returns the next free ring index (gated by that image's
  in-flight fence); `present()` hands the finished image to the encoder queue. Used for
  `RenderMode::Remote` and `RenderMode::Headless`.

**Refactor required:** `renderFrame()` currently calls `vkAcquireNextImageKHR` /
`vkQueuePresentKHR` and references `swapchain.current`, `swapchain.extent`,
`framebuffers.handles[...]` directly. These become calls through `instance.sink`. The ~150
lines of command-buffer recording, the interop timeline logic, and `drawElements` are
unchanged. The depth image, render pass, and framebuffer creation in `initGraphics()`
(`engine.cpp:940`) are reused; only the color image source changes (swapchain images vs.
offscreen ring).

### 4.2 Headless device creation

For a true headless server, create the `VkInstance` / `VkDevice` with **no surface**:
- `createInstance()` (`engine.cpp:849`) must not add the GLFW surface extensions when headless.
  `GlfwContext::getRequiredExtensions()` (`window.hpp:28`) is the source of those — gate it.
- `pickPhysicalDevice` / `findQueueFamilies` (`engine.cpp:740-742`) currently require a surface
  to find a present queue. Headless needs only a graphics (and ideally a dedicated encode)
  queue; the present-queue requirement is dropped.
- No `window_context` / `processEvents()` in headless mode — the loop pulls from
  `ControlSource` instead.

This is the foundational step (see §8, step 1): get the engine to render one frame into an
offscreen `VkImage` with no window.

### 4.3 NVENC encoder

`FrameEncoder` interface so the encoding scheme is swappable:

```cpp
struct EncodedFrame { std::span<const std::byte> data; uint64_t pts; bool keyframe; };

struct FrameEncoder {
    virtual ~FrameEncoder() = default;
    virtual void configure(VkExtent2D extent, VkFormat format) = 0;
    virtual EncodedFrame encode(VkImage image, VkSemaphore renderDone) = 0;
    virtual void requestKeyframe() = 0; // for client reconnect / stream join
};
```

- **`NvencEncoder`** (production path). The rendered image is already on the GPU. Register the
  offscreen `VkImage`'s backing `VkDeviceMemory` as CUDA external memory — the engine already
  does exactly this for buffers in `allocLinear` (`engine.cpp:418`, via
  `interop::importCudaExternalMemory`, `interop.hpp:19`) — and feed it to NVENC through its
  CUDA interface, avoiding any GPU→CPU readback. Codec: **H.264 or HEVC**, low-latency tuning
  (`NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY`), CBR, periodic IDR + on-demand keyframe for client
  join. Target ~10–20 Mbps for 1080p60.
- **`RawEncoder`** (bring-up only). `vkCmdCopyImageToBuffer` to a host-visible buffer, ship
  bytes as-is. Used in step 2 to validate plumbing before NVENC exists. Localhost/LAN only.

NVENC consumes the image on a separate thread; correctness is guaranteed by waiting on the
frame's `renderDone` semaphore before reading. Colorspace: convert RGBA→NV12 (NVENC input)
either in the copy/blit or via a small compute step.

### 4.4 Control application

A `ControlSource` feeds events to the render loop, which applies them to existing state before
each frame — the same mutations the GLFW callbacks perform today:

```cpp
enum class ControlKind { CameraRotate, CameraTranslate, CameraZoom,
                         Pause, Resume, Resize, ToggleView, SetParam, Quit };
struct ControlEvent { ControlKind kind; /* payload: float3 delta, ids, etc. */ };

struct ControlSource {
    virtual ~ControlSource() = default;
    virtual bool poll(ControlEvent& out) = 0; // non-blocking drain
};
```

- **`GlfwControl`** — today's window callbacks (local mode).
- **`RemoteControl`** — drains events decoded from the QUIC control stream.

Mapping to existing API (no new camera math needed):
- `CameraRotate`/`Translate`/`Zoom` → `Camera::rotate` / `Camera::translate` /
  `Camera::setPosition` (`camera.hpp:28-29`), then `updateViewMatrix()`.
- `Pause`/`Resume` → a `bool paused` flag gating the user compute lambda in the loop
  (`display`, `engine.cpp:227`).
- `Resize` → set `window_context.resize_requested`-equivalent; `OffscreenSink` rebuilds its
  ring at the new extent (mirrors `recreateGraphics`, `engine.cpp:1026`), encoder reconfigures,
  client is told the new resolution.
- `ToggleView` → `toggleVisibility` (already in the public API, `mimir.hpp:69`).

## 5. Transport: QUIC

Chosen: **QUIC from the start.** Rationale for this deployment (client dials a directly
addressable server, over the internet):
- One UDP socket with **built-in TLS 1.3** — encryption is mandatory for a public port on a
  university machine, and we get it for free.
- **Congestion control** suited to lossy/variable home links.
- **Multiple independent streams** over one connection: a *video* stream and a *control*
  stream multiplexed without head-of-line blocking between them.
- Because the client initiates to a known `server_ip:port`, we **do not need WebRTC's
  ICE/STUN/TURN** NAT-traversal machinery — that complexity only pays off when both ends are
  behind NAT with no stable address, which is not our case.

Candidate libraries (to evaluate in the design→impl transition): `msquic`, `quiche` (C API),
`lsquic`, or `ngtcp2`. Pick one C/C++ library with a permissive license and integrate via CPM
(`cmake/deps.cmake`), consistent with how Slang/GLFW/etc. are fetched.

### Streams
- **Control stream** (bidirectional, reliable, low bandwidth): client→server `ControlEvent`s;
  server→client session metadata (resolution changes, keyframe markers, metrics). Small
  length-prefixed binary messages.
- **Video stream** (server→client): a header (codec, resolution, SPS/PPS) followed by
  length-prefixed access units. Mark keyframes so a freshly-joined or recovered client can
  start decoding at the next IDR.

### Connection model & deployment (support both)
The client takes a server address and connects out. Two documented deployment paths:
1. **Direct `IP:port`** — server listens on a routable address (cluster head node or an opened
   firewall port). Client connects directly. QUIC's TLS provides encryption end to end.
2. **SSH tunnel** — `ssh -L 9000:localhost:9000 user@cluster`, then the client connects to
   `localhost:9000`. Sidesteps firewall holes and NAT entirely; SSH also encrypts. This is the
   expected default for many cluster users and requires no NAT/firewall config — it's just a
   different address the client dials.

### Transport fallback (decided)
An `ssh -L` tunnel is **TCP**, which defeats QUIC's loss recovery if QUIC packets are forced
through it. Decision: **keep a TCP transport as a fallback** alongside QUIC, selected per
connection. Both implement the same transport interface (see below), so the rest of the
server/client is agnostic:
- **Direct IP:port** → QUIC (preferred): UDP, TLS, congestion control, multiplexed streams.
- **SSH tunnel (or any UDP-hostile path)** → TCP fallback: a single TLS-or-plain TCP connection
  (SSH already encrypts the tunneled case) carrying both video and control as length-prefixed
  framed messages. Accepts TCP's head-of-line blocking as the cost of working everywhere.

Selection: client tries QUIC first and falls back to TCP on failure, or the user forces a mode
via a flag (e.g. `--transport tcp` when connecting through a tunnel). The transport interface:

```cpp
struct Transport {
    virtual ~Transport() = default;
    virtual void sendVideo(std::span<const std::byte> accessUnit, bool keyframe) = 0;
    virtual void sendControl(std::span<const std::byte> msg) = 0;
    virtual bool recvControl(std::vector<std::byte>& out) = 0; // server side
    virtual bool recvVideo(std::vector<std::byte>& out, bool& keyframe) = 0; // client side
};
```
`QuicTransport` maps video/control to separate QUIC streams; `TcpTransport` multiplexes them
over one connection with a small frame header (channel id + length). NVENC/decode, `FrameSink`,
and `ControlSource` sit above this and never know which transport is active.

A pre-stream **handshake/auth** (shared token or similar) gates a connection before any frames
flow. Details TBD; the seam is the control stream's first message.

## 6. Client design (`mimir-client`, new native executable)

A small standalone app (its own target under `samples/` or a top-level `client/`), not linked
into the core library:
- **Args:** `mimir-client <server_addr> <port> [--token ...]`. `server_addr` may be a direct
  IP or a tunneled `localhost`.
- **Connect:** QUIC handshake + auth, then open the control stream and subscribe to video.
- **Decode:** NVDEC (if the client has an NVIDIA GPU) or a portable ffmpeg/CPU decoder so a
  truly thin laptop works. Decode runs on its own thread feeding a small jitter buffer.
- **Display:** blit decoded frames to a window (reuse GLFW; a simple textured quad, or even
  SDL). No CUDA interop needed client-side.
- **Input:** capture mouse/keyboard, translate to `ControlEvent`s, send on the control stream.
  Mirror the server's existing camera control gestures so interaction feels identical to local.

## 7. Module layout & build integration

Keep the new code isolated and **off by default** so the base library's footprint is unchanged:

```
lib/
  include/private/mimir/
    frame_sink.hpp          # FrameSink interface + WindowSink/OffscreenSink
    frame_encoder.hpp       # FrameEncoder interface + RawEncoder/NvencEncoder
    control_source.hpp      # ControlSource interface + GlfwControl/RemoteControl
  src/remote/               # OffscreenSink, NvencEncoder, RemoteControl, QUIC server glue
client/                     # mimir-client native app (decode + display + input)
```

- New `ViewerOptions` field: `enum class RenderMode { Local, Remote, Headless }` (default
  `Local`) plus a nested `RemoteOptions { uint16_t port; std::string bind_addr;
  EncoderConfig encoder; }`. Public API additions are additive — existing constructors and
  samples are unaffected.
- CMake option `MIMIR_ENABLE_REMOTE` (default **OFF**). When ON: fetch the QUIC lib, link the
  NVENC SDK (`CUDA::nvencodeapi` / NVIDIA Video Codec SDK), compile `lib/src/remote/` and the
  `client/` target. When OFF: none of it is built and `mimir` has no new dependencies.
- Selecting `RenderMode::Remote` while `MIMIR_ENABLE_REMOTE=OFF` is a clear runtime/compile
  error.

## 8. Staged implementation plan

Each step is independently testable; the encoder and transport are introduced behind their
interfaces so earlier, simpler implementations are drop-in-replaced.

1. **Headless offscreen render. [DONE]** Implemented as a `RenderMode::Headless` mode-branch
   (not a virtual `FrameSink` yet — deferred to step 3/4 when the encoder needs a second present
   path). Headless device creation (no surface), offscreen color-image ring, render-pass final
   layout `TRANSFER_SRC`, `renderHeadless()` + `saveFrame()` (PPM). Verified: `run_headless`
   renders a point cloud offscreen on GPU.
2. **Raw frame over a socket + minimal client. [DONE]** Raw BGRA frames over TCP
   (`lib/src/remote.cpp::serveRemote`), dependency-free wire protocol
   (`mimir/remote_protocol.hpp`), control-receiver thread feeding the render loop, and a
   standalone thin client (`run_remote_client`). Verified end to end: frames stream, and with
   the sim paused a client camera-rotate changes the rendered view (~42% pixels differ). Server
   sample `run_remote_server` streams a live brownian point cloud. (Client is headless/saves
   PPM for now; a windowed blit client is the next increment.)

   **[DONE]** Windowed blit client `run_remote_viewer` (GLFW + OpenGL, `glDrawPixels`,
   left/right/middle-drag → rotate/zoom/pan, P pause, Q quit). Depends only on the wire
   protocol + GLFW/GL (no mimir/CUDA/Vulkan link), modelling the thin native client.
3. **H.264 encode. [DONE]** Frame payloads are H.264-encoded before sending, behind a
   `remote::Codec` field in the `Hello` handshake (server advertises `RawBGRA` or `H264`, client
   verifies). Implemented with ffmpeg/libav* (`H264Encoder` in `lib/src/remote.cpp`, guarded by
   `MIMIR_HAVE_FFMPEG`): prefers the hardware `h264_nvenc` encoder, falls back to software
   `libx264`; BGRA→YUV420P via libswscale; low-latency tuning per encoder. Gated by the
   `MIMIR_ENABLE_REMOTE` CMake option (pkg-config `libavcodec/libavutil/libswscale`); when the
   library is built without ffmpeg, `serveRemote(..., use_h264=true)` transparently falls back to
   raw. Decoding client `run_remote_decode` (ffmpeg decode → BGRA → PPM). Verified end to end:
   the encode→decode round-trip reconstructs the point cloud, the control round-trip still works
   (sim paused, camera-rotate changes the view), and the stream is **~175–180× smaller** than raw
   (≈20 KB vs 3600 KB per 1280×720 frame). The encode currently still uses the step-1 GPU→CPU
   readback (`readFrameBytes`) before feeding libswscale; the zero-copy CUDA-external-memory path
   into NVENC (avoiding readback) is a later optimisation.
4. **QUIC transport. [DONE]** Done in two parts:
   - *4a — Transport seam.* Extracted `remote::Transport` (`lib/include/private/mimir/transport.hpp`):
     a server-side abstraction with a reliable video channel (Hello + length-prefixed frames) and
     a control channel (ControlMsg). The existing socket code became `TcpTransport`
     (`lib/src/transport_tcp.cpp`); `serveRemote()` gained a `TransportKind` argument and is now
     transport-agnostic. No behavior change on the TCP path.
   - *4b — QUIC.* `QuicTransport` (`lib/src/transport_quic.cpp`) on **ngtcp2** + its OpenSSL
     crypto binding (`ngtcp2_crypto_ossl`, for OpenSSL 3.5+ native QUIC TLS). UDP + TLS 1.3 +
     congestion control; video on a server-initiated uni stream, control on a client-initiated
     uni stream (same byte framing as TCP, since QUIC streams are reliable+ordered). A single I/O
     thread owns the conn and runs the read/write/timer pump (poll + eventfd wakeups), so all
     ngtcp2 calls stay single-threaded. Ephemeral self-signed cert (encryption without auth, like
     a tunnel's first hop). Gated by the `MIMIR_ENABLE_QUIC` CMake option (pkg-config
     `libngtcp2 libngtcp2_crypto_ossl`); without it, `serveRemote(..., Quic)` reports the missing
     support so the caller can fall back to TCP. QUIC thin client `run_remote_quic`. Verified end
     to end over loopback: full TLS handshake, H.264-over-QUIC streams 15 frames at ~180× smaller
     than raw with the control round-trip intact, and raw-over-QUIC (3.6 MB frames) flows through
     stream flow control correctly.

   Still open from the original plan: client-side QUIC→TCP auto-fallback selection (the server
   already speaks both; the bundled clients are per-transport), and the SSH-tunnel path is TCP by
   construction (documented in §5, not separately re-tested here).
5. **Polish.** Auth/handshake, resize handling, keyframe-on-join, reconnect, metrics surfaced
   to the client (reuse `getMetrics`, `mimir.hpp:111`), graceful disconnect.

## 9. Open questions / risks

- ~~QUIC-over-SSH-tunnel~~ **(resolved):** keep a TCP fallback transport behind the same
  `Transport` interface; QUIC for direct connections, TCP for tunneled/UDP-hostile paths. (§5)
- **Decoder portability:** NVDEC assumes an NVIDIA client GPU; ship an ffmpeg/CPU fallback so a
  thin laptop without NVIDIA still works.
- **Colorspace conversion** RGBA→NV12 cost and placement (compute shader vs. NVENC input).
- **Single vs. multi-client** later — the `FrameSink`/transport seams should not preclude
  fan-out, but it's out of scope for v1.
- **Auth model** for a public university port — token now, something stronger later.
```
